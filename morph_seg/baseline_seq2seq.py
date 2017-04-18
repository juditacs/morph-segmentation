from sys import stdin
import tensorflow as tf
import numpy as np
import random
import pandas as pd
from os import path
from sys import argv


class DataSet(object):
    def __init__(self):
        self.vocab_enc = {}
        self.vocab_dec = {}
        self.samples = []
        self.raw_samples = set()
        self.data_enc = None
        self.data_dec = None

    def read_data_from_stream(self, stream, delimiter='', limit=0):
        for line in stream:
            if not line.strip():
                continue
            enc, dec = line.rstrip('\n').split('\t')
            if (enc, dec) in self.raw_samples:
                continue
            self.raw_samples.add((enc, dec))
            if limit > 0 and len(self.raw_samples) > limit:
                break
            if delimiter:
                self.samples.append((enc.split(delimiter), dec.split(delimiter)))
            else:
                self.samples.append((list(enc), list(dec)))
        self.raw_samples = None

    def vectorize_samples(self):
        data_enc = []
        data_dec = []
        self.maxlen_enc = max(len(s[0]) for s in self.samples)
        self.maxlen_dec = max(len(s[1]) for s in self.samples)
        for enc, dec in self.samples:
            padded = ['PAD' for p in range(self.maxlen_enc - len(enc))] + enc
            data_enc.append(
                [self.vocab_enc.setdefault(c, len(self.vocab_enc)) for c in padded]
            )
            padded = ['GO'] + dec + ['PAD' for p in range(self.maxlen_dec - len(dec))]
            data_dec.append(
                [self.vocab_dec.setdefault(c, len(self.vocab_dec)) for c in padded]
            )
        self.maxlen_dec += 1
        self.data_enc = np.array(data_enc)
        self.data_dec = np.array(data_dec)

    def split_train_test(self, test_ratio=.1):
        test_indices = np.random.random(self.data_enc.shape[0])
        mask = test_indices <= test_ratio
        self.data_enc_test = self.data_enc[mask]
        self.data_dec_test = self.data_dec[mask]
        self.data_enc = self.data_enc[~mask]
        self.data_dec = self.data_dec[~mask]

    def get_batch(self, batch_size):
        if self.data_enc is None:
            self.vectorize_samples()
        indices = np.random.choice(self.data_enc.shape[0], batch_size)
        return self.data_enc[indices], self.data_dec[indices]


class SimpleSeq2seq(object):
    def __init__(self, cell_type, cell_size, embedding_size):
        self.init_cell(cell_type, cell_size)
        self.embedding_size = embedding_size

    def init_cell(self, cell_type, cell_size):
        if cell_type == 'LSTM':
            self.cell = tf.contrib.rnn.BasicLSTMCell(cell_size)
        elif cell_type == 'GRU':
            self.cell = tf.contrib.rnn.GRUCell(cell_size)

    def create_model(self, dataset):
        self.create_placeholders(dataset)
        self.initialize_seq2seq(dataset)
        self.create_train_ops()

    def create_placeholders(self, dataset):
        self.enc_inp = [
            tf.placeholder(tf.int32, shape=[None], name='enc{}'.format(i))
                           for i in range(dataset.maxlen_enc)
        ]
        self.dec_inp = [
            tf.placeholder(tf.int32, shape=[None], name='dec{}'.format(i))
                           for i in range(dataset.maxlen_dec)
        ]
        self.weights = [tf.zeros_like(self.dec_inp[0], dtype=tf.float32)]
        self.weights.extend([tf.ones_like(self.dec_inp[i], dtype=tf.float32)
                             for i in range(1, dataset.maxlen_dec)])
        self.feed_previous = tf.placeholder(tf.bool)

    def initialize_seq2seq(self, dataset):
        self.dec_out, self.dec_memory = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
            self.enc_inp, self.dec_inp,
            cell=self.cell,
            num_encoder_symbols=len(dataset.vocab_enc),
            num_decoder_symbols=len(dataset.vocab_dec),
            embedding_size=self.embedding_size,
            dtype=tf.float32,
            feed_previous=self.feed_previous,
        )

    def create_train_ops(self):
        self.loss = tf.contrib.legacy_seq2seq.sequence_loss(
            self.dec_out, self.dec_inp, self.weights
        )
        optimizer = tf.train.MomentumOptimizer(0.05, 0.9)
        self.train_op = optimizer.minimize(self.loss)

    def train_and_test(self, dataset, epochs, batch_size):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            losses = []
            for i in range(epochs):
                batch_enc, batch_dec = dataset.get_batch(batch_size)
                feed_dict = self.populate_feed_dict(batch_enc, batch_dec)
                feed_dict[self.feed_previous] = False
                _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                losses.append(loss)
                if i % 100 == 99:
                    print('Iter {}, loss: {}'.format(i+1, loss))
            test_enc = dataset.data_enc_test
            test_dec = dataset.data_dec_test
            #np.random.shuffle(test_dec)
            feed_dict = self.populate_feed_dict(test_enc, test_dec)
            feed_dict[self.feed_previous] = True
            test_out, loss = sess.run([self.dec_out, self.loss], feed_dict=feed_dict)
            return losses[-1], loss

    def populate_feed_dict(self, batch_enc, batch_dec):
        feed_dict = {}
        for i in range(batch_enc.shape[1]):
            feed_dict[self.enc_inp[i]] = batch_enc[:, i]
        for i in range(batch_dec.shape[1]):
            feed_dict[self.dec_inp[i]] = batch_dec[:, i]
        return feed_dict


def decode_and_print(y, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    decoded = []
    for word in y:
        max_labels = word.argmax(axis=1)
        decoded.append([inv_vocab[m] for m in max_labels])
    seq_len = len(y)
    batch_size = y[0].shape[0]
    print('\n'.join(' '.join(decoded[i][d] for i in range(seq_len)) for d in range(batch_size)))


def main():
    fn = argv[1]
    data = DataSet()
    data.read_data_from_stream(stdin, limit=200000)
    data.vectorize_samples()
    print(data.data_enc.shape)
    data.split_train_test()
    print(data.data_enc.shape)
    if not path.exists(fn):
        df = pd.DataFrame(columns=['cell_type', 'cell_size', 'epochs', 'embedding_size', 'train_loss', 'test_loss'])
    else:
        df = pd.read_table(fn)
    for e in range(100):
        tf.reset_default_graph()
        print('EXPERIMENT {}'.format(e+1))
        cell = random.choice(['LSTM', 'GRU'])
        csize = random.choice([16, 32, 64, 128, 256, 512, 1024])
        embedding_size = random.choice(list(range(2, 30)))
        epochs = [i*100 for i in range(1, 5)]
        epochs = random.choice(epochs)
        res = {
            'cell_type': cell,
            'cell_size': csize,
            'embedding_size': embedding_size,
            'epochs': epochs,
        }
        print(res)
        s = SimpleSeq2seq(cell, csize, embedding_size)
        s.create_model(data)
        train_loss, test_loss = s.train_and_test(data, epochs, 1000)
        res['train_loss'] = train_loss
        res['test_loss'] = test_loss
        print('Train loss: {}, test loss: {}'.format(train_loss, test_loss))
        df = df.append(res, ignore_index=True)
        df.to_csv(fn, sep='\t', index=False)
        
if __name__ == '__main__':
    main()
