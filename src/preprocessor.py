import torch
import json
import pandas as pd
import logging
from multiprocessing import Pool
from dataset import DialogDataset
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
from embedding import Embedding
from transformers import BertTokenizer, XLNetTokenizer, RobertaTokenizer
import re
nltk.download('stopwords')
nltk.download('punkt')

class Preprocessor:
    """

    Args:
        embedding_path (str): Path to the embedding to use.
    """
    def __init__(self, tokenizer):
        super(Preprocessor, self).__init__()
        #self.embedding = embedding
        self.logging = logging.getLogger(name=__name__)
        self.tokenkzer = tokenizer

    def tokenize(self, sentence):
        """ Tokenize a sentence.
        Args:
            sentence (str): One string.
        Return:
            indices (list of str): List of tokens in a sentence.
        """
        # TODO
        token_list = self.tokenizer.tokenize(sentence)
        return token_list
    
    def replaceContraction(self, text):
        contraction_patterns = [ (r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'), (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'), (r'dont', 'do not'), (r'wont', 'will not') ]
        patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
        for (pattern, repl) in patterns:
            (text, count) = re.subn(pattern, repl, text)
        return text

    def sentence_to_indices(self, sentence):
        """ Convert sentence to its word indices.
        Args:
            sentence (str): One string.
        Return:
            indices (list of int): List of word indices.
        """
        token_list = self.tokenize(sentence)
        indices = self.tokenizer.convert_tokens_to_ids(token_list)

        return indices

    def get_dataset(self, data_path, n_workers=4, dataset_args={}):
        """ Load data and return Dataset objects for training and validating.

        Args:
            data_path (str): Path to the data.
            valid_ratio (float): Ratio of the data to used as valid data.
        """
        self.logging.info('loading dataset...')
        dataset = pd.read_csv(data_path)

        self.logging.info('preprocessing data...')

        results = [None] * n_workers
        with Pool(processes=n_workers) as pool:
            for i in range(n_workers):
                batch_start = (len(dataset) // n_workers) * i
                if i == n_workers - 1:
                    batch_end = len(dataset)
                else:
                    batch_end = (len(dataset) // n_workers) * (i + 1)

                batch = dataset[batch_start: batch_end]
                results[i] = pool.apply_async(self.preprocess_samples, [batch])

                # When debugging, you'd better not use multi-thread.
                # results[i] = self.preprocess_dataset(batch, preprocess_args)

            pool.close()
            pool.join()

        processed = []
        for result in results:
            processed += result.get()

        #padding = self.embedding.to_index('[PAD]')
        return DialogDataset(processed, **dataset_args)


    def preprocess_samples(self, dataset):
        """ Worker function.

        Args:
            dataset (list of dict)
        Returns:
            list of processed dict.
        """
        processed = []
        print(len(dataset))
        count = 0
        #self.embedding.add('[CLS]')
        #self.embedding.add('[student]')
        #self.embedding.add('[advisor]')
        #self.embedding.add('[SEP]')
        total_length = 0
        exceed_count = 0
        for i in tqdm(range(len(dataset))):
            sample, length = self.preprocess_sample(dataset.iloc[i, :])
            processed.append(sample)
            total_length += length
            if length > 400:
                exceed_count += 1
            #count += len(self.preprocess_sample(dataset.iloc[i, :])['abstract'])
        print(total_length / len(dataset))
        print(exceed_count / len(dataset))
        return processed

    def label_to_onehot(self, labels):

        label_dict = {'BACKGROUND': 0, 'OBJECTIVES':1, 'METHODS':2, 'RESULTS':3, 'CONCLUSIONS':4, 'OTHERS':5}
        label = []
        onehot = [0, 0, 0, 0, 0, 0]
        for s in labels.split():
            onehot = [0, 0, 0, 0, 0, 0]
            for l in s.split('/'):
                onehot[label_dict[l]] = 1
            label.append(onehot)
        return label

    def remove_stopwords(self, text):
        stoplist = stopwords.words('english')
        finalTokens = []
        tokens = nltk.word_tokenize(text)
        for w in tokens:
            if (w not in stoplist):
                finalTokens.append(w)
        text = " ".join(finalTokens)
        return text

    def preprocess_sample(self, data):
        """
        Args:
            data (dict)
        Returns:
            dict
        """
        processed = {}
        #processed['abstract'].append(self.sentence_to_indices("[CLS]"))
        data['Abstract'] = data['Abstract'].replace('$$$', self.tokenizer.eos_token)
        #data['Abstract'] = data['Abstract'].replace('(i)', '')
        #data['Abstract'] = data['Abstract'].replace('(ii)', '')
        #data['Abstract'] = data['Abstract'].replace('(iii)', '')
        #data['Abstract'] = data['Abstract'].replace('(iv)', '')
        #data['Abstract'] = data['Abstract'].replace('(v)', '')
        data['Abstract'] = self.replaceContraction(data['Abstract'])
        #print(self.tokenizer.encode('In this paper'))
        processed['abstract'] = self.tokenizer.encode(self.tokenizer.bos_token + data['Abstract'] + self.tokenizer.eos_token) 
        #processed['abstract'] = self.tokenizer.encode(data['Abstract'] + self.tokenizer.eos_token)
        #print(processed['abstract'])
        processed['sentence_token'] = []
        processed['sentence_length'] = []
        for i in range(len(processed['abstract'])):
            if self.tokenizer.decode(processed['abstract'][i]) == self.tokenizer.eos_token:
                processed['sentence_token'].append(i)
                if len(processed['sentence_token']) == 1:
                    processed['sentence_length'].append(i)
                else:
                    processed['sentence_length'].append(i - processed['sentence_token'][-2]) 
                if i > 500:
                    print("EXCEED")
        #print(processed['sentence_length']) 
        if 'Task 1' in data:
            processed['label'] = self.label_to_onehot(data['Task 1'])
            #if processed['label'][3] == 1:
            #    print(processed['month'])
            #processed['label'] = self.seq2seq_label(data['Task 2'])
        #print(processed['label'])
        length = len(processed['abstract'])
        return processed, length


