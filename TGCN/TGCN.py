import numpy as np
import pandas as pd
from collections import defaultdict
from genericpath import isfile
import os
from os import listdir
import re
from google.colab import drive
from tqdm import tqdm
import random
import scipy.sparse as sp

import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer

tf.compat.v1.disable_eager_execution()


Random_seed = 2024
NUM_CLASSES = 20
MAX_DOC_LENGTH = 500
unknown_ID = 1
padding_ID = 0
learning_rate = 0.01
num_epochs = 10
batch_size = 60
window_size = 4

random.seed(Random_seed)

class DataReader:
    def __init__(self, data_path, batch_size):
        self._batch_size = batch_size
        with open(data_path) as f:
            d_lines = f.read().splitlines()

        self._data = []
        self._labels = []
        self._sentence_lengths = []
        self._final_tokens = []
        for data_id, line in enumerate(d_lines):
            features = line.split('<fff>')
            label, doc_id, sentence_length = int(features[0]), int(features[1]), int(features[2])
            tokens = features[3].split()

            self._data.append(tokens)
            self._sentence_lengths.append(sentence_length)
            self._labels.append(label)
            self._final_tokens.append(tokens[-1])

        self._data = np.array(self._data)
        self._labels = np.array(self._labels)
        self._sentence_lengths = np.array(self._sentence_lengths)
        self._final_tokens = np.array(self._final_tokens)

        self._num_epoch = 0
        self._batch_id = 0
        self._size = len(self._data)

    def next_batch(self):
        start = self._batch_id * self._batch_size
        end = start + self._batch_size
        self._batch_id += 1

        if end + self._batch_size > len(self._data):
            self._size = end
            end = len(self._data)
            start = end - self._batch_size
            self._num_epoch += 1
            self._batch_id = 0
            indices = list(range(len(self._data)))
            random.shuffle(indices)
            self._data, self._labels, self._sentence_lengths, self._final_tokens = self._data[indices], self._labels[
                indices], self._sentence_lengths[indices], self._final_tokens[indices]

        return self._data[start:end], self._labels[start:end], self._sentence_lengths[start:end], self._final_tokens[
                                                                                                  start:end]



def gen_data_and_vocab():
  def collect_data_from(parent_path, newsgroup_list, word_count=None):
    data = []
    for group_id, newsgroup in enumerate(newsgroup_list):
      dir_path = parent_path + '/' + newsgroup + '/'

      files = [(filename, dir_path + filename)
                for filename in listdir(dir_path)
                if isfile(dir_path + filename)]
      files.sort()
      label = group_id
      print(f'Processing: {group_id}, {newsgroup}')

      for filename, filepath in files:
        with open(filepath, encoding='utf8', errors='ignore') as f:
          text = f.read().lower()
          words = re.split('\W+', text)
          if word_count is not None:
            for word in words:
              word_count[word] += 1
          content = ' '.join(words)
          assert(len(content.splitlines()) == 1)
          data.append(str(label) + '<fff>'
                      + filename + '<fff>' + content)
    return data

  word_count = defaultdict(int)

  path = 'drive/MyDrive/Test_TensorFlow/datasets/20news-bydate/'
  parts = [path + dir_name + '/' for dir_name in listdir(path)
                if not isfile(path + dir_name)]
  if 'train' in parts[0]:
    train_path, test_path = parts[0], parts[1]
  else:
    train_path, test_path = parts[1], parts[0]

  newsgroup_list = [newsgroup for newsgroup in listdir(train_path)]
  newsgroup_list.sort()

  train_data = collect_data_from(
      parent_path=train_path,
      newsgroup_list=newsgroup_list,
      word_count=word_count
  )
  vocab = [word for word, freq in zip(word_count.keys(), word_count.values()) if freq > 10]
  with open('drive/MyDrive/Test_TensorFlow/datasets/w2v/vocab-raw.txt', 'w') as f:
    f.write('\n'.join(vocab))

  test_data = collect_data_from(
      parent_path=test_path,
      newsgroup_list=newsgroup_list
  )

  with open('drive/MyDrive/Test_TensorFlow/datasets/w2v/20news-train-raw.txt', 'w') as f:
    f.write('\n'.join(train_data))

  with open('drive/MyDrive/Test_TensorFlow/datasets/w2v/20news-test-raw.txt', 'w') as f:
    f.write('\n'.join(test_data))

def encode_data(data_path, vocab_path):
  with open(vocab_path) as f:
    vocab = dict([(word, word_ID + 2)
                  for word_ID, word in enumerate(f.read().splitlines())])
  with open (data_path) as f:
    documents = [(line.split('<fff>')[0], line.split('<fff>')[1], line.split('fff')[2])
                  for line in f.read().splitlines()]
  encoded_data = []
  for document in documents:
    label, doc_id, text = document
    words = text.split()[:MAX_DOC_LENGTH]
    sentence_length = len(words)
    encoded_text = []
    for word in words:
      if word in vocab:
        encoded_text.append(str(vocab[word]))
      else:
        encoded_text.append(str(unknown_ID))

    if len(words) < MAX_DOC_LENGTH:
      num_padding = MAX_DOC_LENGTH - len(words)
      for _ in range(num_padding):
        encoded_text.append(str(padding_ID))
    encoded_data.append(f'{label}<fff>{doc_id}<fff>{sentence_length}<fff>'\
             + ' '.join(encoded_text))

  dir_name = '/'.join(data_path.split('/')[:-1])
  file_name = '-'.join(data_path.split('/')[-1].split('-')[:-1]) + '-encoded.txt'
  with open(dir_name + '/' + file_name, 'w') as f:
    f.write('\n'.join(encoded_data))
        

def compute_tfidf(data, vocab_size):
  # tfidf
  data_texts = [' '.join(map(str, row)) for row in data]
  vocab =  [str(i) for i in range(vocab_size + 2)]
  vectorizer = TfidfVectorizer(vocabulary=vocab)
  tfidf_matrix = vectorizer.fit_transform(data_texts)
  '''vectorizer = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True, sublinear_tf=False)
  tfidf_matrix = vectorizer.fit_transform(X).toarray()'''
  return tfidf_matrix


def compute_pmi(X, vocab_size, window_size):
  # PMI
  windows = []
  for doc in X:
    for i in range(len(doc) - window_size + 1):
      windows.append(doc[i:i+window_size])

  word_counts = np.zeros(vocab_size + 2)
  word_pair_counts = np.zeros((vocab_size + 2, vocab_size + 2))
  for window in tqdm(windows):
    for i in range(len(window)):
      word_counts[int(window[i])] += 1
      for j in range(i+1, len(window)):
        word_pair_counts[int(window[i]), int(window[j])] += 1
        word_pair_counts[int(window[j]), int(window[i])] += 1
  epsilon = 1e-8
  print("1 ...")
  matrix_1 = np.outer(word_counts, word_counts) + epsilon
  print("1.5 ...")
  pmi_matrix = np.log((word_pair_counts / matrix_1) + epsilon)
  print("2 ...")
  pmi_matrix[pmi_matrix <= 0] = 0
  print("3 ...")
  return pmi_matrix

def build_adjacency_matrix(data, vocab_size, doc_count, window_size):
  tfidf_matrix = compute_tfidf(data, vocab_size)
  print("Computing tfidf success !!!")
  print(f"tfidf train size = {tfidf_matrix.shape}")
  print("Computing pmi ...")
  pmi_matrix = compute_pmi(data, vocab_size, window_size=window_size)
  print("Computing success !!!")
  A = np.zeros((vocab_size + 2 + doc_count, vocab_size + 2 + doc_count), dtype=np.float32)
  A[:vocab_size + 2, vocab_size + 2:vocab_size + 2 + doc_count] = tfidf_matrix.T.toarray()    #(11314, 18988) => (18988, 11314)  vocab_size = 18986
  A[:vocab_size + 2, :vocab_size + 2] = pmi_matrix
  np.fill_diagonal(A, 1)
  D = np.diag(np.sum(A, axis=1) ** (-0.5))
  A_ = (D * A) * D
  return A_

def build_feature_matrix(vocab_size, doc_count):
  X = np.eye(vocab_size + 2 + doc_count, dtype=np.float32)
  return X

class Text_Graph_Convolutional_Networks(tf.keras.Model):
    def __init__(self,
                 data,
                 vocab_size,
                 doc_count,
                 hidden_dim,
                 num_classes,
                 window_size):
      super(Text_Graph_Convolutional_Networks, self).__init__()
      self._data = data
      #self.A = tf.constant(build_adjacency_matrix(data, vocab_size=vocab_size, doc_count=doc_count, window_size=window_size), dtype=tf.float32)
      #self.X = tf.constant(build_feature_matrix(vocab_size, doc_count), dtype=tf.float32)

      #self.A, self.X = build_adjacency_matrix(data, vocab_size=vocab_size, doc_count=doc_count, window_size=window_size), build_feature_matrix(vocab_size, doc_count)
      self.A = sp.csr_matrix(build_adjacency_matrix(data, vocab_size, doc_count, window_size))
      self.X = build_feature_matrix(vocab_size, doc_count)

      # Nếu không làm như vậy  thì A, X quá lớn => bị lỗi "ValueError: Cannot create a tensor proto whose content is larger than 2GB.""
      

      self.GCN1 = tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu)
      self.GCN2 = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)

    def call(self, inputs):
      #h = self.GCN1(tf.matmul(self.A, self.X))
      #h = tf.concat([self.gcn1(tf.matmul(self.A_blocks[i], self.X_blocks[i])) for i in range(len(self.A_blocks))], axis=0)
      h = self.GCN1(tf.sparse.sparse_dense_matmul(self.A, self.X))
      logits = self.GCN2(h)
      return logits


if __name__ == "__main__":
  drive.mount('/content/drive')
  file_path = 'drive/MyDrive/Test_TensorFlow/datasets/data/words_idfs.txt'
  '''print("Preprocessing ...")
  gen_data_and_vocab()
  print("Preprocessing Success !!!")
  print("Encode Training Data ...")
  encode_data('drive/MyDrive/Test_TensorFlow/datasets/w2v/20news-train-raw.txt', 'drive/MyDrive/Test_TensorFlow/datasets/w2v/vocab-raw.txt')
  print("Encode Training Data Success !!!")
  print("Encode Testing Data ...")
  encode_data('drive/MyDrive/Test_TensorFlow/datasets/w2v/20news-test-raw.txt', 'drive/MyDrive/Test_TensorFlow/datasets/w2v/vocab-raw.txt')
  print("Encode Testing Data !!!")'''

  with open('drive/MyDrive/Test_TensorFlow/datasets/w2v/vocab-raw.txt', encoding='unicode_escape') as f:
    vocab = dict([(word, word_ID+2) for word_ID, word in enumerate(f.read().splitlines())])
  with open('drive/MyDrive/Test_TensorFlow/datasets/w2v/20news-train-encoded.txt', encoding= 'unicode_escape') as f:
    data = f.read().splitlines()
    X_train = np.array([line.split('<fff>')[-1].split(' ') for line in data], dtype=np.float32)
    y_train = np.array([line.split('<fff>')[0] for line in data], dtype=np.int32)
    y_train = tf.one_hot(indices=y_train, depth= NUM_CLASSES, dtype=tf.int32)
  with open('drive/MyDrive/Test_TensorFlow/datasets/w2v/20news-test-encoded.txt', encoding= 'unicode_escape') as f:
    data = f.read().splitlines()
    X_test = np.array([line.split('<fff>')[-1].split(' ') for line in data], dtype=np.float32)
    y_test = np.array([line.split('<fff>')[0] for line in data], dtype=np.int32)
    y_test = tf.one_hot(indices=y_test, depth=NUM_CLASSES, dtype=tf.int32)

  print(f"X_train = {X_train}")
  print(f"y_train = {y_train}")
  print(f"X_test = {X_test}")
  print(f"y_test = {y_test}")

  vocab_size=len(vocab)
  print(f"vocab_size = {vocab_size}")

  model = Text_Graph_Convolutional_Networks(data=X_train,
                                            vocab_size=vocab_size + 2, 
                                            doc_count=X_train.shape[0],
                                            hidden_dim=200,
                                            num_classes=NUM_CLASSES,
                                            window_size=window_size)
  
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=['accuracy'])
  
  model.fit(X_train, y_train, epochs=10, batch_size=batch_size)
  loss, acc = model.evaluate(X_test, y_test)
  print(f"Loss = {loss}, Accuracy = {acc}")

  









