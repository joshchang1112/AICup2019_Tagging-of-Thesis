import torch
from base_predictor import BasePredictor
from modules import BertNet, CategoryNet, Seq2SeqBertNet
from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizer, AdamW

class ExamplePredictor(BasePredictor):
    """

    Args:
        dim_embed (int): Number of dimensions of word embedding.
        dim_hidden (int): Number of dimensions of intermediate
            information embedding.
    """

    def __init__(self, dropout_rate=0.2, loss='BCELoss', margin=0, threshold=None,
                 similarity='inner_product', **kwargs):
        super(ExamplePredictor, self).__init__(**kwargs)
        
        self.model = BertNet()
        #self.model = Seq2SeqBertNet()
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss = torch.nn.BCELoss()
        #self.category_description = torch.load('../data/category_description.pkl')
        #self.loss = torch.nn.MultiLabelSoftMarginLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 4000) 
    def _run_iter(self, batch, training):
        
        context = batch['abstract']
        context_lens = batch['abstract_lens']
        sentence_token = batch['sentence_token']
        label = batch['label']
        '''
        title = batch['title']
        title_lens = batch['title_lens']
        '''
        #id = batch['id']
        #batch_size = context.size()[0]
        '''
        category_description = []
        for i in range(len(category)):
            if len(category) == 0:
                category_description.append(torch.zeros(1, 768))
            for j in range(len(category[i])):
                if j == 0:
                    category_description.append(self.category_description[category[i][j]])
                else:
                    category_description[i] = torch.cat([category_description[i], self.category_description[category[i][j]]], dim=0)
        '''
        #title_description = torch.zeros(batch_size, 768).to(self.device)
        #for i in range(batch_size):
        #    title_description[i] = self.title[id[i]]
        answer_prob = self.model(context.to(self.device), context_lens, sentence_token)
        #answer_prob = self.model(context.to(self.device), context_lens, year.to(self.device), month.to(self.device))
        #answer_prob = self.model(context.to(self.device), context_lens, category.to(self.device), category_lens)
        batch_size = context.size()[0]
        total_sentence = 0
        batch_count = [0] * batch_size
        for i in range(batch_size):
            if i == 0:
                prob = answer_prob[i]
                answer = torch.Tensor(label[i]).float().to(self.device)
                #print(prob.size())
                #print(answer.size())

            else:
                prob = torch.cat([prob, answer_prob[i]], dim=0)
                answer = torch.cat([answer, torch.Tensor(label[i]).float().to(self.device)], dim=0)
                #print(prob.size())
                #print(answer.size())
                #print(sentence_token[i])
        #print(prob.size())
        #print(answer.size())
        loss = self.loss(prob, answer)
                
        return prob, answer, loss
    
    def _predict_batch(self, batch):
    
        context = batch['abstract']
        context_lens = batch['abstract_lens']
        sentence_token = batch['sentence_token']

        answer_prob = self.model(context.to(self.device), context_lens, sentence_token)

        batch_size = context.size()[0]
        total_sentence = 0
        batch_count = [0] * batch_size
        for i in range(batch_size):
            if i == 0:
                prob = answer_prob[i]
                #answer = torch.Tensor(label[i]).float().to(self.device)
                #print(prob.size())
                #print(answer.size())

            else:
                prob = torch.cat([prob, answer_prob[i]], dim=0)
                #answer = torch.cat([answer, torch.Tensor(label[i]).float().to(self.device)], dim=0)
        return prob
