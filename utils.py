import tensorflow as tf
import numpy as np
import pickle


def get_shape(tensor):
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims


def count_parameters(trained_vars):
  total_parameters = 0
  print('=' * 100)
  for variable in trained_vars:
    variable_parameters = 1
    for dim in variable.get_shape():
      variable_parameters *= dim.value
    print('{:70} {:20} params'.format(variable.name, variable_parameters))
    print('-' * 100)
    total_parameters += variable_parameters
  print('=' * 100)
  print("Total trainable parameters: %d" % total_parameters)
  print('=' * 100)


def read_vocab(vocab_file):
  print('Loading vocabulary ...')
  with open(vocab_file, 'rb') as f:
    word_to_index = pickle.load(f)
    print('Vocabulary size = %d' % len(word_to_index))
    return word_to_index


def batch_doc_normalize(docs):
  sent_lengths = np.array([len(doc) for doc in docs], dtype=np.int32)
  max_sent_length = sent_lengths.max()
  word_lengths = [[len(sent) for sent in doc] for doc in docs]
  max_word_length = max(map(max, word_lengths))

  padded_docs = np.zeros(shape=[len(docs), max_sent_length, max_word_length], dtype=np.int32)  # PADDING 0
  word_lengths = np.zeros(shape=[len(docs), max_sent_length], dtype=np.int32)
  for i, doc in enumerate(docs):
    for j, sent in enumerate(doc):
      word_lengths[i, j] = len(sent)
      for k, word in enumerate(sent):
        padded_docs[i, j, k] = word

  return padded_docs, sent_lengths, max_sent_length, word_lengths, max_word_length


def load_glove(glove_file, emb_size, vocab): #vocab는 dictionary 형태이다. 
  print('Loading Glove pre-trained word embeddings ...')
  embedding_weights = {}
  f = open(glove_file, encoding='utf-8')
  for line in f:
    values = line.split() #단어와 embedding size 개수 만큼의 숫자가 들어있음
    word = values[0] #단어 추출 
    vector = np.asarray(values[1:], dtype='float32') #embedding vector 추출 
    embedding_weights[word] = vector #dictionary화 
  f.close()
  print('Total {} word vectors in {}'.format(len(embedding_weights), glove_file))
  #-0.5에서 0.5사이의 값 랜덤으로 생성. output shape = (len(vocab), emb_size) 
  embedding_matrix = np.random.uniform(-0.5, 0.5, (len(vocab), emb_size)) / emb_size

  oov_count = 0
  for word, i in vocab.items():
    embedding_vector = embedding_weights.get(word)
    if embedding_vector is not None: #데이터에 등장한 단어가 glove 단어 사전에 존재하면 임베딩 값을 할당. 
      embedding_matrix[i] = embedding_vector
    else: #그렇지 않으면 oov 카운트, 랜덤 값을 사용한다. 
      oov_count += 1
  print('Number of OOV words = %d' % oov_count)

  return embedding_matrix
