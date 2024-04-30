# -*- coding: gbk -*-
# @Time : 2023/3/15 20:55
# @Author : wufei
# @File : aspect_finetune_MAMS.py


import pickle

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import get_linear_schedule_with_warmup
from sklearn import metrics
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from transformers import AdamW, BertConfig, BertForSequenceClassification
from transformers import BertPreTrainedModel
from model.module.bert_post_ln import BertMLMHead, BertPostLayerNormalizationModel, ABSAOutput
from model.module.bert_pre_ln import BertPreLayerNormalizationModel
from torch.nn import CrossEntropyLoss
import os
from torch import nn, optim
import torch.nn.functional as F


class ABSADataset(Dataset):
    def __init__(self, path):
        super(ABSADataset, self).__init__()
        data = pickle.load(open(path, 'rb'))
        self.raw_texts = data['raw_texts']
        self.raw_aspect_terms = data['raw_aspect_terms']
        self.bert_tokens = [torch.LongTensor(
            token) for token in data['bert_tokens']]
        self.aspect_mask = [torch.FloatTensor(
            mask) for mask in data['aspect_masks']]
        self.implicits = torch.LongTensor(data['implicits'])
        self.labels = torch.LongTensor(data['labels'])
        self.len = len(data['labels'])

    def __getitem__(self, index):
        return (self.bert_tokens[index],
                self.aspect_mask[index],
                self.labels[index],
                self.raw_texts[index],
                self.raw_aspect_terms[index],
                self.implicits[index])

    def __len__(self):
        return self.len


def collate_fn(batch):
    bert_tokens, aspect_masks, labels, raw_texts, raw_aspect_terms, implicits = zip(*batch)

    bert_masks = pad_sequence([torch.ones(tokens.shape) for tokens in bert_tokens], batch_first=True)
    bert_tokens = pad_sequence(bert_tokens, batch_first=True)
    aspect_masks = pad_sequence(aspect_masks, batch_first=True)
    labels = torch.stack(labels)
    implicits = torch.stack(implicits)

    return (bert_tokens,
            bert_masks,
            aspect_masks,
            labels,
            raw_texts,
            raw_aspect_terms,
            implicits)


def prepare_dataset(train_path, test_path, absa_dataset, collate_fn=default_collate, batch_size=16, num_works=0):
    train_loader = DataLoader(absa_dataset(train_path),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_works,
                              collate_fn=collate_fn) if train_path else None
    test_loader = DataLoader(absa_dataset(test_path),
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_works,
                             collate_fn=collate_fn) if test_path else None
    return train_loader, test_loader


import pickle



class SCAPT(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler", 'cls.predictions.bias',
                                          'cls.predictions.transform.dense.weight',
                                          'cls.predictions.transform.dense.bias',
                                          'cls.predictions.decoder.weight',
                                          'cls.seq_relationship.weight',
                                          'cls.seq_relationship.bias',
                                          'cls.predictions.transform.LayerNorm.weight',
                                          'cls.predictions.transform.LayerNorm.bias']
    _keys_to_ignore_on_load_missing = [r"position_ids", r"decoder.bias", r"classifier",
                                       'cls.bias', 'cls.transform.dense.weight',
                                       'cls.transform.dense.bias', 'cls.transform.LayerNorm.weight',
                                       'cls.transform.LayerNorm.bias', 'cls.decoder.weight',
                                       'cls_representation.weight', 'cls_representation.bias',
                                       'aspect_representation.weight', 'aspect_representation.bias']

    def __init__(self, config, hidden_size=256):
        super().__init__(config)

        if config.model == 'BERT':
            self.bert = BertPostLayerNormalizationModel(config, add_pooling_layer=False)
        elif config.model == 'TransEnc':
            self.bert = BertPreLayerNormalizationModel(config, add_pooling_layer=False)
        else:
            raise TypeError(f"Not supported model {config['model']}")
        self.cls = BertMLMHead(config)

        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_representation = nn.Linear(config.hidden_size, hidden_size)
        self.aspect_representation = nn.Linear(config.hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(2 * hidden_size, config.num_labels)
        self.init_weights()

        self.CoAttention = CoAttention()

    def forward(
            self,
            input_ids=None,
            index=None,
            implicits=None,
            attention_mask=None,
            aspect_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            multi_card=False,
            has_opposite_labels=False,
            pretrain_ce=False
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs, entity_embedding, inputs_embeds = self.bert(
            input_ids,
            index,
            aspect_masks=aspect_mask,
            implicits=implicits,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.CoAttention(sequence_output, entity_embedding, inputs_embeds, aspect_mask, implicits)

        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        expand_aspect_mask = (1 - aspect_mask).unsqueeze(-1).bool()
        cls_hidden = self.cls_representation(sequence_output[:, 0])
        aspect_hidden = torch.div(torch.sum(sequence_output.masked_fill(expand_aspect_mask, 0), dim=-2),
                                  torch.sum(aspect_mask.float(), dim=-1).unsqueeze(-1))
        aspect_hidden = self.aspect_representation(aspect_hidden)
        merged = self.dropout(torch.cat((cls_hidden, aspect_hidden), dim=-1))
        sentiment = self.classifier(merged)
        if multi_card:
            if has_opposite_labels:
                return cls_hidden, outputs.last_hidden_state[:, 0], masked_lm_loss
            else:
                return outputs.last_hidden_state[:, 0], masked_lm_loss
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return sentiment, cls_hidden, masked_lm_loss, output
        return ABSAOutput(
            sentiment=sentiment,
            loss=masked_lm_loss,
            cls_hidden=cls_hidden,
            logits=prediction_scores,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def build_absa_model(config, embedding_layer=None):
    bert_config = BertConfig.from_pretrained("bert-base-uncased")
    bert_config.num_labels = 3
    bert_config.hidden_dropout_prob = config['dropout']
    bert_config.id2label = {
        0: 'positive',
        1: 'negative',
        2: 'neutral'
    }
    bert_config.label2id = {
        'positive': 0,
        'negative': 1,
        'neutral': 2,
    }
    bert_config.model = config['model']
    bert_for_facts_absa = SCAPT.from_pretrained('bert-base-uncased', config=bert_config)
    return bert_for_facts_absa


def evaluate(config, model, data_loader, criterion=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_samples, correct_samples = 0, 0
    total_explicit, correct_explicit = 0, 0
    total_implicit, correct_implicit = 0, 0
    total_loss = 0
    model.eval()
    labels_all, preds_all = None, None

    with torch.no_grad():
        for batch in data_loader:
            bert_tokens, bert_masks, aspect_masks, labels, raw_texts, raw_aspect_terms, implicits = batch
            bert_tokens = bert_tokens.to(device)
            bert_masks = bert_masks.to(device)
            aspect_masks = aspect_masks.to(device)
            labels = labels.to(device)
            implicits = implicits.to(device)
            index = cal_index(aspect_masks)
            index = index.to(device)

            output = model(
                input_ids=bert_tokens,
                attention_mask=bert_masks,
                aspect_mask=aspect_masks,
                index=index,
                implicits=implicits,
                output_attentions=True,
                return_dict=True
            )

            logits = output.sentiment
            loss = criterion(logits, labels) if criterion else 0
            batch_size = bert_tokens.size(0)
            if loss > 0:
                total_loss += batch_size * loss.item()
            total_samples += batch_size
            pred = logits.argmax(dim=-1)
            if labels_all is None:
                labels_all = labels
                preds_all = pred
            else:
                labels_all = torch.cat((labels_all, labels), dim=0)
                preds_all = torch.cat((preds_all, pred), dim=0)
            correct_samples += (pred == labels).long().sum().item()
            total_explicit += (1 - implicits).long().sum().item()
            correct_explicit += ((1 - implicits) & (pred == labels)).long().sum().item()
            total_implicit += implicits.long().sum().item()
            correct_implicit += (implicits & (pred == labels)).long().sum().item()

            # del bert_tokens, bert_masks, aspect_masks, index, location, output, logits, loss, pred, labels
            # torch.cuda.empty_cache()

    accuracy = correct_samples / total_samples
    f1 = metrics.f1_score(y_true=labels_all.cpu(),
                          y_pred=preds_all.cpu(),
                          labels=[0, 1, 2], average='macro')
    average_loss = total_loss / total_samples if criterion else 0.0
    explicit_acc = correct_explicit / total_explicit if total_explicit else 0.0
    implicit_acc = correct_implicit / total_implicit if total_implicit else 0.0
    return accuracy, f1, average_loss, explicit_acc, implicit_acc


def build_optimizer(config, model):
    lr = config['learning_rate']
    weight_decay = config['weight_decay']
    opt = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'adamw': AdamW,
        'adagrad': optim.Adagrad,
    }
    if 'momentum' in config:
        optimizer = opt[config['optimizer']](
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=config['momentum']
        )
    else:
        optimizer = opt[config['optimizer']](
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    return optimizer


class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


def cal_index(aspect_masks):
    index = torch.zeros([aspect_masks.shape[0], 1])
    index_pre = torch.nonzero(aspect_masks == 1)
    for index_num in range(aspect_masks.shape[0]):
        index_sum = 0
        index_length = 0
        location_ = []
        for i in range(index_pre.shape[0]):
            if index_pre[i][0] == index_num:
                index_sum += index_pre[i][1]
                index_length += 1
                location_.append(int(index_pre[i][1]))
        index[index_num][0] = index_sum / index_length
    return index


def aspect_finetune(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = prepare_dataset(config['train_file'], config['test_file'], ABSADataset, collate_fn,
                                                config['batch_size'], config['num_workers_per_loader'])
    model = build_absa_model(config).to(device)
    state_dict = config.get('checkpoint', None)
    if isinstance(state_dict, str):
        model.load_state_dict(torch.load(state_dict))
    elif state_dict:
        model.load_state_dict(state_dict)

    if train_loader is None:
        val_acc, val_f1, _, explicit_acc, implicit_acc = evaluate(config, model, test_loader)
        print("valid f1: {:.4f}, valid acc: {:.4f}, explicit acc: {:.4f}, implicits acc: {:.4f}".format(val_f1, val_acc,
                                                                                                        explicit_acc,
                                                                                                        implicit_acc))
        return

    optimizer = build_optimizer(config, model)
    scheduler = get_linear_schedule_with_warmup(optimizer, config['warm_up'], config['epoch'] * len(train_loader))
    classification_criterion = LabelSmoothLoss(smoothing=config['label_smooth'])

    max_val_accuracy = max_val_f1 = 0.
    min_val_loss = float('inf')
    max_explicit_acc = 0.
    max_implicit_acc = 0.
    global_step = 0

    for epoch in range(config['epoch']):
        total_loss = 0.
        total_samples = 0
        correct_samples = 0
        for idx, batch in enumerate(train_loader):
            global_step += 1
            model.train()

            bert_tokens, bert_masks, aspect_masks, labels, raw_texts, raw_aspect_terms, implicits = batch
            bert_tokens = bert_tokens.to(device)
            bert_masks = bert_masks.to(device)
            aspect_masks = aspect_masks.to(device)
            labels = labels.to(device)
            index = cal_index(aspect_masks)
            index = index.to(device)
            implicits = implicits.to(device)

            output = model(
                input_ids=bert_tokens,
                attention_mask=bert_masks,
                aspect_mask=aspect_masks,
                index=index,
                implicits=implicits,
                output_attentions=True,
                return_dict=True
            )

            logits = output.sentiment
            classification_loss = classification_criterion(logits, labels)
            classification_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm'])

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            batch_size = bert_tokens.size(0)
            total_loss += batch_size * classification_loss.item()
            total_samples += batch_size
            pred = logits.argmax(dim=-1)
            correct_samples += (pred == labels).long().sum().item()
            valid_frequency = config['valid_frequency'] if 'valid_frequency' in config else len(train_loader) - 1

            if idx % valid_frequency == 0 and idx != 0:
                train_loss = total_loss / total_samples
                train_accuracy = correct_samples / total_samples
                total_loss = total_samples = correct_samples = 0
                val_acc, val_f1, val_loss, explicit_acc, implicit_acc = evaluate(config, model, test_loader,
                                                                                 classification_criterion)
                print("[Epoch {:2d}] [step {:3d}]".format(epoch, idx),
                      "train loss: {:.4f}, train acc: {:.4f}, ".format(train_loss, train_accuracy),
                      "valid loss: {:.4f}, valid f1: {:.4f}, valid acc: {:.4f}, ".format(val_loss, val_f1, val_acc),
                      "valid explicit acc: {:.4f}, valid implicit acc: {:.4f} ".format(explicit_acc, implicit_acc))

                max_val_f1 = max(max_val_f1, val_f1)
                if val_acc > max_val_accuracy:
                    max_val_accuracy = val_acc
                    min_val_loss = val_loss
                    max_explicit_acc = explicit_acc
                    max_implicit_acc = implicit_acc

                    model_file = "epoch_{}_step_{}_acc_{:.4f}_f1_{:.4f}_loss_{:.4f}.pt".format(
                        epoch, global_step, val_acc, val_f1, val_loss)
                    # torch.save(model.state_dict(), os.path.join(model_path, model_file))

            # del bert_tokens, bert_masks, aspect_masks, index, location, output, logits, classification_loss, pred, labels
            # torch.cuda.empty_cache()

    print("Max valid accuracy: {:.4f}, valid f1: {:.4f}, ".format(max_val_accuracy, max_val_f1),
          "explicit acc: {:.4f}, implicit acc: {:.4f}".format(max_explicit_acc, max_implicit_acc))
    return max_val_accuracy, max_val_f1, min_val_loss


import random
import numpy as np


def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


if __name__ == '__main__':
    config = {'mode': 'aspect_finetune', 'model': 'BERT', 'device': 0, 'data_path': 'data/laptops',
              'model_path': 'results/laptops',
              'train_file': '/ISA/Data/MAMS/train_preprocess_finetune.pkl',
              'test_file': '/ISA/Data/MAMS/test_preprocess_finetune.pkl',
              'num_workers_per_loader': 16,
              'seed': 42, 'batch_size': 32, 'dropout': 0.1, 'epoch': 8, 'warm_up': 680, 'label_smooth': 0.06,
              'optimizer': 'adagrad', 'learning_rate': 5e-05, 'weight_decay': 0.015, 'grad_norm': 6.0,
              'checkpoint': '/ISA/Data/BERT_MAMS.pt'}
    seed_torch(config['seed'])
    aspect_finetune(config)

    