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
            tokens = int(features[3].split())

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

class Text_Graph_Convolutional_Networks(tf.keras.Model):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 num_classes,
                 window_size,
                 batch_size):
        super(Text_Graph_Convolutional_Networks, self).__init__()

        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._batch_size = batch_size
        self._num_classes = num_classes
        self._window_size = window_size

        self._data = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, MAX_DOC_LENGTH])
        self._labels = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, ])
        self._tfidf_matrix = tf.compat.v1.placeholder(tf.float32, shape=[None, None])
        self._pmi_matrix = tf.compat.v1.placeholder(tf.float32, shape=[None, None])

        self.A = tf.compat.v1.placeholder(tf.float32, shape=[None, None])
        self.X = tf.compat.v1.placeholder(tf.float32, shape=[None, None])


        #self.A, self.X = self.build_graph()
        #self.logits = self.trainer()

    '''def build_graph(self):
      tfidf_matrix = self._tfidf_matrix #

      pmi_matrix = self._pmi_matrix #

      A = np.zeros((self._vocab_size + 2 + self._data.shape[0], self._vocab_size + 2 + self._data.shape[0]))

      for i in range(2, self._vocab_size + 2):
        for j in range(self._vocab_size + 2, self._vocab_size + 2 + self._data.shape[0]):
          A[i, j] = tfidf_matrix[j - self._vocab_size, i]

      A[2:self._vocab_size + 2, 2:self._vocab_size + 2] = pmi_matrix
      A += np.eye(A.shape[0])

      D = np.diag(np.sum(A, axis=1) ** (-0.5))
      A_ = D @ A @ D

      X = np.eye(self._vocab_size + 2 + self._data.shape[0], dtype=np.float32)

      return tf.constant(A, dtype=tf.float32), tf.constant(X, dtype=tf.float32) # Ko train'''

    def trainer(self, learning_rate):
      #W0 = tf.compat.v1.variable(tf.random.normal([self._vocab_size + 2 + self._data.shape[0], self._embedding_size]), name='W0')
      W0 = tf.compat.v1.get_variable(
          name='W0_18',
          shape=(self._vocab_size + 2 + self._data.shape[0], self._embedding_size),
          initializer=tf.random_normal_initializer(seed=Random_seed)
          #reuse=tf.compat.v1.AUTO_REUSE
      )
      #W1 = tf.compat.v1.variable(tf.random.normal([self._embedding_size, self._num_classes]), name='W1')
      W1 = tf.compat.v1.get_variable(
          name='W1_18',
          shape=(self._embedding_size, self._num_classes),
          initializer=tf.random_normal_initializer(seed=Random_seed)
          #reuse=tf.compat.v1.AUTO_REUSE
      )

      h1 = tf.nn.relu(tf.matmul(self.normalize_adjacency_matrix(self.A), tf.matmul(self.X, W0)))

      logits = tf.matmul(self.normalize_adjacency_matrix(self.A), h1) @ W1
      predict = tf.nn.softmax(logits)


      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._labels, logits=logits))
      # Add Regularization ...

      optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
      train_op = optimizer.minimize(loss)

      return predict, loss, train_op
    def normalize_adjacency_matrix(self, A):
        D = tf.compat.v1.diag(tf.pow(tf.reduce_sum(A, axis=1), -0.5))
        return D @ A @ D

def compute_tfidf(X, vocab_size):
  # tfidf
  X_texts = [' '.join(map(str, row)) for row in X]
  vocab =  [str(i) for i in range(vocab_size + 2)]
  vectorizer = TfidfVectorizer(vocabulary=vocab)
  tfidf_matrix = vectorizer.fit_transform(X_texts)
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
  epsilon = 1e-3
  print("1 ...")
  pmi_matrix = np.log((word_pair_counts / (word_counts[:, None] * word_counts[None, :] + epsilon)) + epsilon)
  print("2 ...")
  pmi_matrix[pmi_matrix <= 0] = 0
  print("3 ...")
  return pmi_matrix


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

  model = Text_Graph_Convolutional_Networks(vocab_size=vocab_size,
                                          embedding_size=100,
                                          num_classes=NUM_CLASSES,
                                          window_size=window_size,
                                          batch_size=batch_size)
  #A, X = model.build_graph()
  predicted_labels, loss, train_op = model.trainer(learning_rate=learning_rate)


  with tf.compat.v1.Session() as sess:
      train_data_reader = DataReader(
          data_path = 'drive/MyDrive/Test_TensorFlow/datasets/w2v/20news-train-encoded.txt',
          batch_size=batch_size
      )
      test_data_reader = DataReader(
          data_path = 'drive/MyDrive/Test_TensorFlow/datasets/w2v/20news-test-encoded.txt',
          batch_size=batch_size
      )
      step = 0
      MAX_STEP = 5000
      sess.run(tf.compat.v1.global_variables_initializer())
      for step in tqdm(range(MAX_STEP)):
        next_train_batch = train_data_reader.next_batch()
        train_data, train_labels, train_sentence_lengths, train_final_tokens = next_train_batch

        print("Computing tfidf train ...")
        tfidf_matrix_train = compute_tfidf(train_data, vocab_size)
        print("Computing success !!!")
        print(f"tfidf train size = {tfidf_matrix_train.shape}")
        print("Computin pmi train ...")
        pmi_matrix_train = compute_pmi(train_data, vocab_size, window_size=window_size)
        print("Computing success")
        A_train = np.zeros((vocab_size + 2 + tfidf_matrix_train.shape[0], vocab_size + 2 + tfidf_matrix_train.shape[0]), dtype=np.float32)
        print(f"SHape 1 = {A_train[:vocab_size + 2, vocab_size + 2:vocab_size + 2 + tfidf_matrix_train.shape[0]].shape}")
        print(f"SHape 2 = {tfidf_matrix_train.T.shape}")
        A_train[:vocab_size + 2, vocab_size + 2:vocab_size + 2 + tfidf_matrix_train.shape[0]] = tfidf_matrix_train.T.toarray()    #(11314, 18988) => (18988, 11314)  vocab_size = 18986
        A_train[:vocab_size + 2, :vocab_size + 2] = pmi_matrix_train
        np.fill_diagonal(A_train, 1)
        D_train = np.diag(np.sum(A_train, axis=1) ** (-0.5))
        A_train_ = (D_train * A_train) * D_train
        X1_train = np.eye(vocab_size + 2 + tfidf_matrix_train.shape[0], dtype=np.float32)

        print(f"train_data {train_data.dtype}")
        print(f"labels {train_labels.dtype}")
        print(f"tfidf {tfidf_matrix_train.dtype}")
        print(f"pmi {pmi_matrix_train.dtype}")
        print(f"A_train {A_train.dtype}")
        print(f"X1_train {X1_train.dtype}")
        plabels_eval, loss_eval, _ = sess.run(
            [predicted_labels, loss, train_op],
            feed_dict={
                model._data: train_data,
                model._labels: train_labels,
                model._tfidf_matrix: tfidf_matrix_train,
                model._pmi_matrix: pmi_matrix_train,
                model.A: A_train,
                model.X: X1_train
            }
        )
        if step % 50 == 0:
          print("predicted labels = ")
          print(plabels_eval)
          print("train labels = ")
          print(train_labels)
          print(f" Loss = {loss_eval}, Batch id = {train_data_reader._batch_id}")
        if train_data_reader._batch_id == 0:
          num_true_preds = 0
          while True:
            next_test_batch = test_data_reader.next_batch()
            test_data, test_labels, test_sentence_lengths, test_final_tokens = next_test_batch

            print("Computing tfidf test ...")
            tfidf_matrix_test = compute_tfidf(test_data, vocab_size)
            print("Computing success !!!")
            print(f"tfidf test size = {tfidf_matrix_test.shape}")
            print("Computin pmi test ...")
            pmi_matrix_test = compute_pmi(test_data, vocab_size, window_size=window_size)
            print("Computing success")
            A_test = np.zeros((vocab_size + 2 + tfidf_matrix_test.shape[0], vocab_size + 2 + tfidf_matrix_test.shape[0]), dtype=np.float32)
            print(f"SHape 1 = {A_test[:vocab_size + 2, vocab_size + 2:vocab_size + 2 + tfidf_matrix_test.shape[0]].shape}")
            print(f"SHape 2 = {tfidf_matrix_test.T.shape}")
            A_test[:vocab_size + 2, vocab_size + 2:vocab_size + 2 + tfidf_matrix_test.shape[0]] = tfidf_matrix_test.T.toarray()    #(11314, 18988) => (18988, 11314)  vocab_size = 18986
            A_test[:vocab_size + 2, :vocab_size + 2] = pmi_matrix_test
            np.fill_diagonal(A_test, 1)
            D_test = np.diag(np.sum(A_test, axis=1) ** (-0.5))
            A_test_ = (D_test * A_test) * D_test
            X1_test = np.eye(vocab_size + 2 + tfidf_matrix_test.shape[0], dtype=np.float32)

            test_plabels_eval = sess.run(
                predicted_labels,
                feed_dict={
                    model._data: test_data,
                    model._labels: test_labels,
                    model._tfidf_matrix: tfidf_matrix_test,
                    model._pmi_matrix: pmi_matrix_test,
                    model.A: A_test,
                    model.X: X1_test
                }
            )
            print("test plabels eval = ")
            print(test_plabels_eval)
            print("test labels = ")
            print(test_labels)
            matches=np.equal(test_plabels_eval, test_labels)
            num_true_preds += np.sum(matches.astype(float))

            if test_data_reader._batch_id == 0:
              break
          print(f"Epoch : {train_data_reader._num_epoch}, num true predicts = {num_true_preds}, length test data = {len(test_data_reader._data)}, Accuracy on test data = {num_true_preds * 100. / len(test_data_reader._data)}")









