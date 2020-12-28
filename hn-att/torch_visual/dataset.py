import os
import csv

import pandas as pd
import torch
from torch.utils.data import Dataset
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.datasets import fetch_20newsgroups

#torch.utils.data.Dataset 은 데이터셋을 나타내는 추상클래스입니다. 여러분의 데이터셋은 Dataset 에 상속하고 아래와 같이 오버라이드 해야합니다.
class News20Dataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, train_file, word_map_path, max_sent_length=100, max_doc_length=10, is_train=True):
        """
        :param cache_data_path: folder where data files are stored
        :param word_map_path: path for vocab dict, used for embedding
        :param max_sent_length: maximum number of words in a sentence
        :param max_doc_length: maximum number of sentences in a document 
        :param is_train: true if TRAIN mode, false if TEST mode
        """
        self.max_sent_length = max_sent_length
        self.max_doc_length = max_doc_length
        self.split = 'train' if is_train else 'test'
        
        self.train_file=train_file
        self.data = pd.read_csv(train_file, sep='\t')
        
#         self.data = fetch_20newsgroups(
#             data_home=cache_data_dir,
#             subset=self.split,
#             categories=['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'],
#             shuffle=False,
#             remove=('headers', 'footers', 'quotes'))

        # glove 파일 기반 
        # 인덱스 0 :<pad>, 인덱스 1: <unk>인 단어 사전 형성  
        self.vocab = pd.read_csv(
            filepath_or_buffer=word_map_path,
            header=None,
            sep=" ",
            quoting=csv.QUOTE_NONE,
            usecols=[0]).values[:50000]
        self.vocab = ['<pad>', '<unk>'] + [word[0] for word in self.vocab]

    # NOTE MODIFICATION (REFACTOR)
    #
    def transform(self, text):
        # encode document
        # doc: 문장 단위로 분리 후 단어 단위로 분리, 각 단어를 vocab의 인덱스로 변환. 
        # vocab에 포함되어 있지 않은 단어는 1(<unk>)로 등록함. 
        # unk으로 변환된 특수 단어를 미리 단어사전의 포함하는 시도가 필요 
        doc = [
            [self.vocab.index(word) if word in self.vocab else 1 for word in word_tokenize(text=sent)]
            for sent in sent_tokenize(text=text)]  # if len(sent) > 0
        doc = [sent[:self.max_sent_length] for sent in doc][:self.max_doc_length]
        
        num_sents = min(len(doc), self.max_doc_length)
        # skip erroneous ones
        if num_sents == 0:
            return None, -1, None

        num_words = [min(len(sent), self.max_sent_length) for sent in doc][:self.max_doc_length]

        return doc, num_sents, num_words

    def __getitem__(self, i):
        label = self.data['label'][i]
        text = self.data['text'][i]

        # NOTE MODIFICATION (REFACTOR)
        doc, num_sents, num_words = self.transform(text)

        if num_sents == -1:
            return None

        return doc, label, num_sents, num_words

    def __len__(self):
        return len(self.data['text'])

    @property
    def vocab_size(self):
        return len(self.vocab)
    
    @property
    def num_classes(self):
        return 3
        # return len(list(self.data.target_names))


def collate_fn(batch):
    batch = filter(lambda x: x is not None, batch)
    docs, labels, doc_lengths, sent_lengths = list(zip(*batch))

    bsz = len(labels)
    batch_max_doc_length = max(doc_lengths)
    batch_max_sent_length = max([max(sl) if sl else 0 for sl in sent_lengths])

    docs_tensor = torch.zeros((bsz, batch_max_doc_length, batch_max_sent_length)).long()
    sent_lengths_tensor = torch.zeros((bsz, batch_max_doc_length)).long()

    for doc_idx, doc in enumerate(docs):
        doc_length = doc_lengths[doc_idx]
        sent_lengths_tensor[doc_idx, :doc_length] = torch.LongTensor(sent_lengths[doc_idx])
        for sent_idx, sent in enumerate(doc):
            sent_length = sent_lengths[doc_idx][sent_idx]
            docs_tensor[doc_idx, sent_idx, :sent_length] = torch.LongTensor(sent)

    return docs_tensor, torch.LongTensor(labels), torch.LongTensor(doc_lengths), sent_lengths_tensor
    
