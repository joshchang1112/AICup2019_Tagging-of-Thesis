import torch


class Metrics:
    def __init__(self):
        self.name = 'Metric Name'

    def reset(self):
        pass

    def update(self, predicts, batch):
        pass

    def get_score(self):
        pass


class Recall(Metrics):
    """
    Args:
         ats (int): @ to eval.
         rank_na (bool): whether to consider no answer.
    """
    def __init__(self):
        self.n = 0
        #self.threshold = [0.43, 0.38, 0.4, 0.4, 0.36, 0.4]
        self.threshold = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
        self.true_positives = [0] * 7
        self.false_positives = [0] * 7
        self.true_negatives = [0] * 7
        self.false_negatives = [0] * 7
        
        self.name = 'F1_Score: '

    def reset(self):
        self.n = 0
        for i in range(7):
            self.true_positives[i] = 0
            self.false_positives[i] = 0
            self.true_negatives[i] = 0
            self.false_negatives[i] = 0

    def update(self, predicts, label):
        """
        Args:
            predicts (FloatTensor): with size (batch, n_samples).
            batch (dict): batch.
        """
        predicts = predicts.cpu()
        batch_size = list(predicts.size())[0]
        n_class = list(predicts.size())[1]
        #print(n_class)
        #print(predicts)
        #print(predicts.size())
        #print(batch['label'].size())
        
        #predict_label = torch.zeros((batch_size, n_class + 1))
        predict_label = torch.zeros((batch_size, n_class))
        for i in range(batch_size):
            tmp = 0
            for j in range(n_class):
                if predicts[i][j] >= self.threshold[j]:
                    predict_label[i][j] = 1
                    tmp = 1
                else:
                    predict_label[i][j] = 0
            
            if tmp == 0:
                _, max_id = torch.max(predicts[i].unsqueeze(0), dim=1)
                #print(max_id)
                predict_label[i][max_id] = 1
                #predict_label[i][3] = 1
            
        for i in range(batch_size):
            for j in range(n_class):
                if predict_label[i][j] == label[i][j].cpu():
                    if predict_label[i][j] == 1:
                        self.true_positives[j] += 1
                        self.true_positives[6] += 1
                    else:
                        self.true_negatives[j] += 1
                        self.true_negatives[6] += 1
                else:
                    if predict_label[i][j] == 1:
                        self.false_positives[j] += 1
                        self.false_positives[6] += 1
                    else:
                        self.false_negatives[j] += 1
                        self.false_negatives[6] += 1

    def get_f1(self):
        recall = self.true_positives[6] / (self.true_positives[6] + self.false_negatives[6] + 1e-20)
        precision = self.true_positives[6] / (self.true_positives[6] + self.false_positives[6] + 1e-20)
        f1_score = (2 * precision * recall) / (precision + recall + 1e-20)
        return f1_score
    
    def get_category_f1(self):
        recall = [0] * 6
        precision = [0] * 6
        f1_score = [0] * 6
        for i in range(6):
            recall[i] = self.true_positives[i] / (self.true_positives[i] + self.false_negatives[i] + 1e-20)
            precision[i] = self.true_positives[i] / (self.true_positives[i] + self.false_positives[i] + 1e-20)
            f1_score[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-20)
            #print(f1_score)
        return f1_score
        #return f1_score
    
    def print_score(self):
        f1 = self.get_f1()
        #self.get_category_f1()
        return '{:.3f}'.format(f1)

