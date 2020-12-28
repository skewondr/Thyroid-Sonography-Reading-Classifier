import os
import torch
from tqdm import tqdm
from dataset import collate_fn
from pylab import *
from nltk.tokenize import word_tokenize, sent_tokenize

import matplotlib
import matplotlib.pyplot as plt

# NOTE MODIFICATION (REFACTOR)
class MetricTracker(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, summed_val, n=1):
        self.val = summed_val / n
        self.sum += summed_val
        self.count += n
        self.avg = self.sum / self.count


# NOTE MODIFICATION (EMBEDDING)
def get_pretrained_weights(glove_path, corpus_vocab, embed_dim, device):
    """
    Returns 50002 words' pretrained weights in tensor
    :param glove_path: path of the glove txt file
    :param corpus_vocab: vocabulary from dataset
    :return: tensor (len(vocab), embed_dim)
    """
    save_dir = os.path.join(glove_path, 'glove_pretrained.pt')
    if os.path.exists(save_dir):
        return torch.load(save_dir, map_location=device)

    corpus_set = set(corpus_vocab)
    pretrained_vocab = set()
    glove_pretrained = torch.zeros(len(corpus_vocab), embed_dim)
    with open(os.path.join(glove_path, 'glove.6B.100d.txt'), 'rb') as f:
        for l in tqdm(f):
            line = l.decode().split()
            if line[0] in corpus_set:
                pretrained_vocab.add(line[0])
                glove_pretrained[corpus_vocab.index(line[0])] = torch.from_numpy(np.array(line[1:]).astype(np.float))

        # handling 'out of vocabulary' words
        var = float(torch.var(glove_pretrained))
        for oov in corpus_set.difference(pretrained_vocab):
            glove_pretrained[corpus_vocab.index(oov)] = torch.empty(100).float().uniform_(-var, var)
        print("weight size:", glove_pretrained.size())
        torch.save(glove_pretrained, save_dir)
    return glove_pretrained


# NOTE MODIFICATION (FEATURE)
# referenced to https://gist.github.com/ihsgnef/f13c35cd46624c8f458a4d23589ac768
def map_sentence_to_color(words, scores, sent_score):
    """
    :param words: array of words
    :param scores: array of attention scores each corresponding to a word
    :param sent_score: sentence attention score
    :return: html formatted string
    """

    sentencemap = matplotlib.cm.get_cmap('Blues')
    wordmap = matplotlib.cm.get_cmap('Oranges')
    result = '<p><span style="margin:5px; padding:5px; background-color: {}">'\
        .format(matplotlib.colors.rgb2hex(sentencemap(sent_score)[:3]))
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    for word, score in zip(words, scores):
        color = matplotlib.colors.rgb2hex(wordmap(score)[:3])
        result += template.format(color, '&nbsp' + word + '&nbsp')
    result += '</span><p>'
    return result


# NOTE MODIFICATION (FEATURE)
#한 example에 대한 모델 예측 결과(softmax 확률)를 bar chart로 출력 
def bar_chart(categories, scores, graph_title='Prediction', output_name='prediction_bar_chart.png'):
    y_pos = arange(len(categories))

    plt.bar(y_pos, scores, align='center', alpha=0.5)
    plt.xticks(y_pos, categories)
    plt.ylabel('Attention Score')
    plt.title(graph_title)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.savefig(output_name)

def visualize_chart(model, dataset, doc):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # transform(doc) 한 example에 대한 결과 출력. 
    doc, num_sents, num_words = dataset.transform(doc[0])
    label = 0  # dummy label for transformation

    doc, label, doc_length, sent_length = collate_fn([(doc, label, num_sents, num_words)])
    
    score, word_att_weight, sentence_att_weight \
        = model(doc.to(device), doc_length.to(device), sent_length.to(device))
    
    # predicted = int(torch.max(score, dim=1)[1])
    classes = ['0','1','2']
#     classes = ['Cryptography', 'Electronics', 'Medical', 'Space']
    result = "<h2>Attention Visualization</h2>"

    bar_chart(classes, torch.softmax(score.detach(), dim=1).flatten().cpu(), 'Prediction')
    result += '<br><img src="prediction_bar_chart.png"><br>'
    
    return result 

def visualize_doc(model, dataset, doc, answer):
    # 입력된 doc을 사전에 학습된 모델에 넣고 weight 시각화  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    # Predicts, and visualizes one document with html file
    :param model: pretrained model
    :param dataset: news20 dataset
    :param doc: document to feed in
    :return: html formatted string for whole document
    """
    #문장 분리 후 단어 분리 
    orig_doc = [word_tokenize(sent) for sent in sent_tokenize(doc)]
    # doc: 
    doc, num_sents, num_words = dataset.transform(doc)
    label = 0  # dummy label for transformation

    doc, label, doc_length, sent_length = collate_fn([(doc, label, num_sents, num_words)])
    
    score, word_att_weight, sentence_att_weight \
        = model(doc.to(device), doc_length.to(device), sent_length.to(device))
    
    predict = torch.argmax(score.detach(), dim=1).flatten().cpu()
    
    if predict==answer: #모델이 답을 맞춘 경우 
        result = "<p>Examples of correct prediction results:</p>"
        result+= '<input type="text" name="serial" value="%s" >' %(answer)

    elif predict!=answer:#모델이 답을 틀린 경우 
        result = "<p>Examples of wrong prediction results:</p>"
        result+= '<input type="text" name="serial" value="%s" >' %(predict)
        result+= '<input type="text" name="serial" value="%s" >' %(answer)
        
    for orig_sent, att_weight, sent_weight in zip(orig_doc, word_att_weight[0].tolist(), sentence_att_weight[0].tolist()):
        result += map_sentence_to_color(orig_sent, att_weight, sent_weight)

    return result
    
# # NOTE MODIFICATION (FEATURE)
# def visualize(model, dataset, doc):
#     # 입력된 doc을 사전에 학습된 모델에 넣고 weight 시각화  
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     """
#     # Predicts, and visualizes one document with html file
#     :param model: pretrained model
#     :param dataset: news20 dataset
#     :param doc: document to feed in
#     :return: html formatted string for whole document
#     """
#     #문장 분리 후 단어 분리 
#     orig_doc = [word_tokenize(sent) for sent in sent_tokenize(doc)]
#     # doc: 
#     doc, num_sents, num_words = dataset.transform(doc)
#     label = 0  # dummy label for transformation

#     doc, label, doc_length, sent_length = collate_fn([(doc, label, num_sents, num_words)])
#     score, word_att_weight, sentence_att_weight \
#         = model(doc.to(device), doc_length.to(device), sent_length.to(device))

#     # predicted = int(torch.max(score, dim=1)[1])
#     classes = ['0','1','2']
# #     classes = ['Cryptography', 'Electronics', 'Medical', 'Space']
#     result = "<h2>Attention Visualization</h2>"

#     bar_chart(classes, torch.softmax(score.detach(), dim=1).flatten().cpu(), 'Prediction')
#     result += '<br><img src="prediction_bar_chart.png"><br>'
    
#     for orig_sent, att_weight, sent_weight in zip(orig_doc, word_att_weight[0].tolist(), sentence_att_weight[0].tolist()):
#         result += map_sentence_to_color(orig_sent, att_weight, sent_weight)

#     return result

