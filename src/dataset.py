import random
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, BertTokenizer, XLNetTokenizer, GPT2Tokenizer

class DialogDataset(Dataset):
    """
    Args:
        data (list): List of samples.
        padding (int): Index used to pad sequences to the same length.
        n_negative (int): Number of false options used as negative samples to
            train. Set to -1 to use all false options.
        n_positive (int): Number of true options used as positive samples to
            train. Set to -1 to use all true options.
        shuffle (bool): Do not shuffle options when sampling.
            **SHOULD BE FALSE WHEN TESTING**
    """
    def __init__(self, data, padding="<pad>", context_padded_len=500, shuffle=True):
        self.data = data
        self.context_padded_len = context_padded_len
        #self.padding = padding
        self.shuffle = shuffle
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        #self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        #self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
        #self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.padding = self.tokenizer.pad_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = dict(self.data[index])
        if len(data['abstract']) > self.context_padded_len:
            data['abstract'] = data['abstract'][:500]
        return data

    def collate_fn(self, datas):
        batch = {}
        # collate lists
        #batch['id'] = [data['id'] for data in datas]
        batch['abstract_lens'] = [len(data['abstract']) for data in datas]
        padded_len = min(self.context_padded_len, max(batch['abstract_lens']))

        batch['abstract'] = torch.tensor(
            [pad_to_len(data['abstract'], padded_len, self.tokenizer, self.padding)
             for data in datas]
        )
        batch['label'] = [data['label'] for data in datas]
        batch['sentence_token'] = [data['sentence_token'] for data in datas]
        
        return batch


def pad_to_len(arr, padded_len, tokenizer, padding="<pad>"):
    """ Pad `arr` to `padded_len` with padding if `len(arr) < padded_len`.
    If `len(arr) > padded_len`, truncate arr to `padded_len`.
    Example:
        pad_to_len([1, 2, 3], 5, -1) == [1, 2, 3, -1, -1]
        pad_to_len([1, 2, 3, 4, 5, 6], 5, -1) == [1, 2, 3, 4, 5]
    Args:
        arr (list): List of int.
        padded_len (int)
        padding (int): Integer used to pad.
    """
    # TODO
    length_arr = len(arr)
    new_arr = arr

    if length_arr < padded_len:
        for i in range(padded_len - length_arr):
            if padding == 5:
                new_arr.append(padding)
            else:
                new_arr.extend(tokenizer.encode(tokenizer.pad_token))
                #new_arr.append(551)
    else:
        for i in range(length_arr - padded_len):
            del new_arr[-1]
    #print(len(new_arr))
    return new_arr


