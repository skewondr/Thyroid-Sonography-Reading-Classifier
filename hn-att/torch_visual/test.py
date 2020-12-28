from dataset import News20Dataset, collate_fn
from utils import *
import os, sys
import webbrowser
import pandas as pd
from sklearn.metrics import classification_report
import argparse
import itertools

def saving(out, labels,test_preds, test_y): 
    test_preds.append(out)
    test_y.append(labels)
    return test_preds, test_y

def flatten(test_preds, test_y):
    preds = [l.tolist() for l in test_preds]
    y = [l.tolist() for l in test_y]
#     print("preds:",preds)
#     print("y:",y)
    final_preds = [item for sublist in preds for item in sublist]
    final_y = [item for sublist in y for item in sublist]
#     print("final_preds:",final_preds)
#     print("final_y:",final_y)
    return final_preds, final_y

def evaluate(prediction, target):
    print("prediction:",prediction)
    print("target:",target)
    report = classification_report(target, prediction)
    print(report)
            
class Tester:
    def __init__(self, config, model, flag='eval'):
        self.config = config
        self.model = model
        self.device = next(self.model.parameters()).device
        
        if flag=='eval':
            train_file = config.cache_data_dir[config.dir_index]+'dev.tsv'
            print("Start Validation...")
        else :
            train_file = config.cache_data_dir[config.dir_index]+'test.tsv'
            print("Start Testing...")
            
        self.dataset = News20Dataset(train_file, config.vocab_path, is_train=False)
        
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=config.batch_size, shuffle=False,
                                                      collate_fn=collate_fn)

        self.accs = MetricTracker()
        self.best_acc = 0
        
    def eval(self):
        self.model.eval()
        with torch.no_grad():
            self.accs.reset()

            for (docs, labels, doc_lengths, sent_lengths) in self.dataloader:
                batch_size = labels.size(0)

                docs = docs.to(self.device)
                labels = labels.to(self.device)
                doc_lengths = doc_lengths.to(self.device)
                sent_lengths = sent_lengths.to(self.device)

                scores, word_att_weights, sentence_att_weights = self.model(docs, doc_lengths, sent_lengths)

                predictions = scores.max(dim=1)[1]
                correct_predictions = torch.eq(predictions, labels).sum().item()
                acc = correct_predictions

                self.accs.update(acc, batch_size)
            self.best_acc = max(self.best_acc, self.accs.avg)

            print('Validation Average Accuracy: {acc.avg:.4f}'.format(acc=self.accs))
            
    def test(self):
        self.model.eval()
        with torch.no_grad():
            self.accs.reset()
            
            test_preds = []
            test_y=[]
            for (docs, labels, doc_lengths, sent_lengths) in self.dataloader:
                batch_size = labels.size(0)

                docs = docs.to(self.device)
                labels = labels.to(self.device)
                doc_lengths = doc_lengths.to(self.device)
                sent_lengths = sent_lengths.to(self.device)
                
                scores, word_att_weights, sentence_att_weights = self.model(docs, doc_lengths, sent_lengths)
                #print("scores:",scores)
                predictions = scores.max(dim=1)[1]
                correct_predictions = torch.eq(predictions, labels).sum().item()
                acc = correct_predictions
                #print("acc:",acc)
                #print("row_result:",predictions,"label:", labels)
                test_preds, test_y = saving(predictions.cpu().detach().numpy(), labels.cpu().detach().numpy(), test_preds, test_y)

                self.accs.update(acc, batch_size)
            self.best_acc = max(self.best_acc, self.accs.avg)
            
                    
            print('Test Average Accuracy: {acc.avg:.4f}'.format(acc=self.accs))
            fin_preds, fin_y = flatten(test_preds, test_y)
            evaluate(fin_preds, fin_y)
            
if __name__ == "__main__":
    
    jump_string = "../../"
    data_dir_list = [jump_string+"../data/uneven-even/",jump_string+"../data/even-even/",
                jump_string+"../data/upeven-even/"]
    output_dir_list =  ["./hn-att_result/uneven-even/","./hn-att_result/even-even/",
                    "./hn-att_result/upeven-even/"]
    
    parser = argparse.ArgumentParser(description='Bug squash for Hierarchical Attention Networks')
    #terminal에서 python train.py --dir_index = ?? 라고 입력하면 됨. 
    parser.add_argument("--dir_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    
    parser.add_argument("--vocab_path", type=str, default="data/glove/glove.6B.100d.txt")
    parser.add_argument("--cache_data_dir", type=str, default=data_dir_list)
    parser.add_argument("--cache_model_dir", type=str, default=output_dir_list)

    config = parser.parse_args()
    
#     if not os.path.exists(output_dir_list[dir_index]+"best_model/model.pth.tar"):
#         print("Visualization requires pretrained model to be saved under ./best_model.\n")
#         print("Please run 'python train.py <args>'")
#         sys.exit()

    checkpoint = torch.load(config.cache_model_dir[config.dir_index]+"best_model/model.pth.tar")
    model = checkpoint['model']
    model.eval()

    dataset = News20Dataset(config.cache_data_dir[config.dir_index]+'test.tsv', "data/glove/glove.6B.100d.txt", is_train=False)
    
    test_doc = pd.read_csv(config.cache_data_dir[config.dir_index]
                           +'test.tsv', sep='\t')
    docs = test_doc['text']
    answers =  test_doc['label']
#     doc = "First of all, realize that Tesla invented AC power generators, motors,\
#     transformers, conductors, etc. Technically, *ALL* transformers are Tesla\
#     coils.  In general though when someone refers to a Tesla coil, they mean\
#     an 'air core resonant transformers'."

#"""한 example에 대한 바 차트 보고싶으면 아래 코드 활성화 하기!"""
#     result = visualize_chart(model, dataset, docs)
#     with open('result.html', 'w') as f:
#         f.write(result)
   
    for doc, answer in zip(docs,answers):     
        result = visualize_doc(model, dataset, doc, answer)
        with open(config.cache_model_dir[config.dir_index]+'result.html', 'a') as f:
            f.write(result)

    #webbrowser.open_new(config.cache_model_dir[config.dir_index]+'result.html')
    
    tester = Tester(config, model, 'test').test()
