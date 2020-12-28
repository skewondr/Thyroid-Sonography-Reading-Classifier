#코드 출처: https://github.com/sharkmir1/Hierarchical-Attention-Network
import os, sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import HierarchicalAttentionNetwork
from dataset import News20Dataset
from dataloader import MyDataLoader
from trainer import Trainer
from utils import get_pretrained_weights

import nltk
nltk.download('punkt')
import sys 

def train(config, device):
    train_file = config.cache_data_dir[config.dir_index]+'train.tsv'
  
    dataset = News20Dataset(train_file, config.vocab_path, is_train=True)

    dataloader = MyDataLoader(dataset, config.batch_size)

    model = HierarchicalAttentionNetwork(
        num_classes=dataset.num_classes, #클래스 개수 
        vocab_size=dataset.vocab_size,#단어 사전 크기 
        embed_dim=config.embed_dim,#단어 임베딩 차원~glove 벡터 차원과 연관됨. 
        word_gru_hidden_dim=config.word_gru_hidden_dim,#단어 레벨 gru 은닉층 차원
        sent_gru_hidden_dim=config.sent_gru_hidden_dim,#문장 레벨 gru 은닉층 차원 
        word_gru_num_layers=config.word_gru_num_layers,
        sent_gru_num_layers=config.sent_gru_num_layers,
        word_att_dim=config.word_att_dim,
        sent_att_dim=config.sent_att_dim,
        use_layer_norm=config.use_layer_norm,
        dropout=config.dropout).to(device)

    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)

    # NOTE MODIFICATION (BUG)
    # criterion = nn.NLLLoss(reduction='sum').to(device) # option 1
    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)  # option 2

    # NOTE MODIFICATION (EMBEDDING)
    if config.pretrain:
        weights = get_pretrained_weights("data/glove", dataset.vocab, config.embed_dim, device)
        model.sent_attention.word_attention.init_embeddings(weights)
    model.sent_attention.word_attention.freeze_embeddings(config.freeze)

    trainer = Trainer(config, model, optimizer, criterion, dataloader, config.cache_model_dir[config.dir_index])
    trainer.train()


if __name__ == '__main__':
    
    jump_string = "../../"
    data_dir_list = [jump_string+"../data/uneven-even/",jump_string+"../data/even-even/",
                jump_string+"../data/upeven-even/"]
    output_dir_list =  ["./hn-att_result/uneven-even/","./hn-att_result/even-even/",
                    "./hn-att_result/upeven-even/"]
    
    parser = argparse.ArgumentParser(description='Bug squash for Hierarchical Attention Networks')
    #terminal에서 python train.py --dir_index = ?? 라고 입력하면 됨. 
    parser.add_argument("--dir_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--max_grad_norm", type=float, default=5)

    parser.add_argument("--embed_dim", type=int, default=100) #glove dim 바뀔때마다 바꿔주기 
    parser.add_argument("--word_gru_hidden_dim", type=int, default=100)
    parser.add_argument("--sent_gru_hidden_dim", type=int, default=100)
    parser.add_argument("--word_gru_num_layers", type=int, default=1)
    parser.add_argument("--sent_gru_num_layers", type=int, default=1)
    parser.add_argument("--word_att_dim", type=int, default=200)
    parser.add_argument("--sent_att_dim", type=int, default=200)
    
    parser.add_argument("--vocab_path", type=str, default="data/glove/glove.6B.100d.txt")
    parser.add_argument("--cache_data_dir", type=str, default=data_dir_list)
    parser.add_argument("--cache_model_dir", type=str, default=output_dir_list)

    # NOTE MODIFICATION (EMBEDDING)
    parser.add_argument("--pretrain", type=bool, default=True)
    parser.add_argument("--freeze", type=bool, default=False)

    # NOTE MODIFICATION (FEATURES)
    parser.add_argument("--use_layer_norm", type=bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.1)

    config = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Make necessary data directories at the very first run
#     if not os.path.exists(os.path.dirname(config.vocab_path)):
#         for dir in [os.path.dirname(config.vocab_path), config.cache_data_dir]:
#             os.makedirs(dir, exist_ok=True)
#         print("Finished making data directories.")
#         print("Before proceeding, please put the GloVe text file under data/glove as instructed.")
#         print("Ending this run.")
#         sys.exit()

#     # NOTE MODIFICATION (FEATURE)
#     if not os.path.exists(os.path.dirname('best_model')):
#         os.makedirs('best_model', exist_ok=True)

    train(config, device)
