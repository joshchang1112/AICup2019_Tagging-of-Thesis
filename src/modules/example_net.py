import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, XLNetModel, GPT2Model 
import scipy.stats
import random

class RobertaForMultiLabelSequenceClassification(torch.nn.Module):
    """XLNET model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, num_labels=4):
        super(RobertaForMultiLabelSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.classifier = torch.nn.Linear(768 , 6)
        self.bert = RobertaModel.from_pretrained('roberta-base', output_hidden_states=True)
        #self.bert = BertModel.from_pretrained('bert-base-cased')
        #self.bert = XLNetModel.from_pretrained('xlnet-large-cased', output_hidden_states=True)
        #self.bert = GPT2Model.from_pretrained('gpt2')
        #self.bert = RobertaModel.from_pretrained('roberta-large', output_hidden_states=True)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.dropout3 = torch.nn.Dropout(0.5)
        self.dropout4 = torch.nn.Dropout(0.5)
        self.dropout5 = torch.nn.Dropout(0.5)
        self.sigmoid = torch.nn.Sigmoid()
        SEED = 0
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        #self.apply(self.init_bert_weights)
        self.device = torch.device('cuda:3' if torch.cuda.is_available()
                                       else 'cpu')
    
    def forward(self, input_ids, context_lens, sentence_token, token_type_ids=None, attention_mask=None, labels=None):
        _, _, last_hidden_state = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #last_hidden_state = self.pool_hidden_state(last_hidden_state, context_lens)
        #last_hidden_state = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #last_hidden_state = last_hidden_state[1]
        #print(last_hidden_state.size())
        last_hidden_state = last_hidden_state[12] + last_hidden_state[11]
        #last_hidden_state = (last_hidden_state[24] + last_hidden_state[23] + last_hidden_state[22] + last_hidden_state[21]) / 4
        #last_hidden_state = torch.cat([last_hidden_state[12], last_hidden_state[11]], dim=2)
        batch_size = input_ids.size()[0]
        logits = []
        for i in range(batch_size):
            for j in range(len(sentence_token[i])):
                if sentence_token[i][j] > 500:
                    if j == len(sentence_token[i]) - 1:
                        tmp_prob = torch.Tensor([0.1, 0.1, 0.1, 0.3, 0.8, 0.01]).unsqueeze(0).to(self.device)
                        prob = torch.cat([prob, tmp_prob], dim=0)
                    else:
                        tmp_prob = torch.Tensor([0.1, 0.1, 0.1, 0.8, 0.3, 0.01]).unsqueeze(0).to(self.device)
                        prob = torch.cat([prob, tmp_prob], dim=0)

                else:
                    if j == 0:
                        logit_mean = torch.mean(last_hidden_state[i][:sentence_token[i][j]+1].unsqueeze(0), 1)
                        prob = self.sigmoid(self.classifier(logit_mean))
                    else:
                        logit_mean = torch.mean(last_hidden_state[i][sentence_token[i][j-1]+1:sentence_token[i][j]+1].unsqueeze(0), 1)
                        prob = torch.cat([prob, self.sigmoid(self.classifier(logit_mean))], dim=0)
                        #prob = torch.cat([prob, self.sigmoid(self.classifier(torch.mean(last_hidden_state[i][sentence_token[i][j-1]+1:sentence_token[i][j]+1].unsqueeze(0), 1)))], dim=0)
                         
            logits.append(prob)
        '''
        logits1 = self.classifier(self.dropout1(last_hidden_state))
        logits2 = self.classifier(self.dropout2(last_hidden_state))
        logits3 = self.classifier(self.dropout3(last_hidden_state))
        logits4 = self.classifier(self.dropout4(last_hidden_state))
        logits5 = self.classifier(self.dropout5(last_hidden_state))
        
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        '''
        #logits = self.classifier(last_hidden_state)
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits

    def pool_hidden_state(self, last_hidden_state, context_lens):
        """
        Pool the output vectors into a single mean vector 
        """
        #mean_last_hidden_state, _ = torch.max(last_hidden_state, 1)
        last_hidden_state = last_hidden_state[0]
        #batch_size = last_hidden_state.size()[0]
        #mean_last_hidden_state = torch.zeros(batch_size, 768).to(self.device)
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        #print(mean_last_hidden_state.size())
        return mean_last_hidden_state

class BertNet(torch.nn.Module):

    def __init__(self):
        super(BertNet, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()
        #self.model = BertForMultiLabelSequenceClassification.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.model = RobertaForMultiLabelSequenceClassification()
        SEED = 0
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        
        '''       
        freeze_layers = 10
        for p in self.model.bert.embeddings.parameters():
            p.requires_grad = False
        self.model.bert.embeddings.dropout.p = 0.0
        for idx in range(freeze_layers):
            for p in self.model.bert.encoder.layer[idx].parameters():
                p.requires_grad = False
            self.model.bert.encoder.layer[idx].attention.self.dropout.p = 0.0
            self.model.bert.encoder.layer[idx].attention.output.dropout.p = 0.0
            self.model.bert.encoder.layer[idx].output.dropout.p = 0.0
        '''

        self.device = torch.device('cuda:3' if torch.cuda.is_available()
                                       else 'cpu')


    def forward(self, context, context_lens, sentence_token):
        batch_size = context.size()[0]
        max_context_len = context.size()[1]

        padding_mask = []
        for i in range(batch_size):
            tmp = [1] * context_lens[i] + [0] * (max_context_len - context_lens[i])
            padding_mask.append(tmp)

        padding_mask = torch.Tensor(padding_mask).to(self.device)
        
        answer_prob = self.model(context, context_lens, sentence_token, attention_mask=padding_mask)
        #answer_prob = self.sigmoid(answer_prob)
        return answer_prob
