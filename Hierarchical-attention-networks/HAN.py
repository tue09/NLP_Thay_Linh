import numpy as np
import pandas as pd
from collections import defaultdict
from genericpath import isfile
import os
from os import listdir
import re
from tqdm import tqdm
import random

import tensorflow as tf
tf.compat.v1.disable_eager_execution()


Random_seed = 2024
NUM_CLASSES = 20
MAX_DOC_LENGTH = 500
unknown_ID = 1
padding_ID = 0
learning_rate = 0.01
num_epochs = 10

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


class Hierarchical_Attention_Networks(tf.keras.Model):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 embedding_dim,
                 gru_units,
                 num_classes,
                 batch_size):
        super(Hierarchical_Attention_Networks, self).__init__()

        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._embedding_dim = embedding_dim
        self._gru_units = gru_units
        self._batch_size = batch_size

        self._data = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, MAX_DOC_LENGTH])
        self._labels = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, ])
        self._sentence_lengths = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, ])
        self._final_tokens = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, ])

        self.embedding = tf.keras.layers.Embedding(input_dim = vocab_size + 2, output_dim = embedding_dim, input_length=MAX_DOC_LENGTH, mask_zero=True)

        self.word_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_units, return_sequences=True))
        self.word_attention = tf.keras.layers.Dense(1, activation='tanh')

        self.sentence_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_units, return_sequences=True))
        self.sentence_attention = tf.keras.layers.Dense(1, activation='tanh')

        self.classifier = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')

    def build_graph(self):
        word_embedding = self.embedding(self._data)
        word_annotation = self.word_gru(word_embedding)
        word_weight = tf.nn.softmax(self.word_attention(word_annotation), axis=1)
        sentence_vector = tf.reduce_sum(word_annotation * word_weight, axis=1)
        print(f"sentence_vector before = {sentence_vector}")
        sentence_vector = tf.expand_dims(sentence_vector, axis=1)
        print(f"sentence_vector after = {sentence_vector}")
        sentence_annotation = self.sentence_gru(sentence_vector)
        sentence_weight = tf.nn.softmax(self.sentence_attention(sentence_annotation), axis=1)
        document_vector = tf.reduce_sum(sentence_annotation * sentence_weight, axis=1)
        logits = self.classifier(document_vector)
        one_hot_labels = tf.one_hot(indices=self._labels, depth=NUM_CLASSES)
        print(f"logits = {logits}")
        print(f"logits shape = {logits.shape}")
        print(f"labels = {self._labels}")
        print(f"labels shape = {self._labels.shape}")
        #print(f"one hot shape = {one_hot_labels.shape}")


        #loss = tf.reduce_mean(tf.nn.log_poisson_loss(logits=logits, labels=self._labels))
        predicted_labels = tf.argmax(logits, axis=1)
        print(f"predict shape before = {predicted_labels.shape}")
        print(f"predict dtype before = {predicted_labels.dtype}")
        predicted_labels = tf.squeeze(predicted_labels)
        print(f"labels shape = {self._labels.shape}")
        print(f"predict data type = {predicted_labels.dtype}")
        print(f"labels data type = {self._labels.dtype}")
        loss = tf.keras.losses.CategoricalCrossentropy()(one_hot_labels, logits)
        #loss = tf.keras.losses.SparseCategoricalCrossentropy()(self._labels, logits)
        return logits, predicted_labels, loss

    def trainer(self, loss, learning_rate):
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
        return optimizer

def evaluate(model, X_data, y_data):
    predictions, loss = model.build_graph()
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_data, axis=1), tf.argmax(predictions, axis=1)), tf.float32))
    return loss, accuracy



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
  model = Hierarchical_Attention_Networks(vocab_size=len(vocab),
                                          embedding_size=100,
                                          embedding_dim=200,
                                          gru_units=50,
                                          num_classes=NUM_CLASSES,
                                          batch_size=batch_size)
  logits, predicted_labels, loss = model.build_graph()
  train_op = model.trainer(loss=loss, learning_rate=learning_rate)
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
      logits_val, plabels_eval, loss_eval, _ = sess.run(
          [logits, predicted_labels, loss, train_op],
          feed_dict={
              model._data: train_data,
              model._labels: train_labels,
              model._sentence_lengths: train_sentence_lengths,
              model._final_tokens: train_final_tokens
          }
      )
      if step % 50 == 0:
        print("logits = ")
        print(logits_val)
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
          test_plabels_eval = sess.run(
              predicted_labels,
              feed_dict={
                  model._data: test_data,
                  model._labels: test_labels,
                  model._sentence_lengths: test_sentence_lengths,
                  model._final_tokens: test_final_tokens
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




