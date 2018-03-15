from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
import csv

from record_reader import TextLoader
from model import inference_graph, loss_graph

flags = tf.flags

# data
flags.DEFINE_string('data_dir',    'data',   'data directory')
flags.DEFINE_string('train_dir',   'check_rnn_single',     'training directory')

# model params
flags.DEFINE_integer('rnn_size',        725,        'size of LSTM internal state')
flags.DEFINE_integer('char_embed_size', 325,        'dimensionality of character embeddings')
flags.DEFINE_integer('rnn_layers',      1,          'number of layers in the LSTM')
flags.DEFINE_integer('num_highway_layer', 0,            'number of highway layer')
flags.DEFINE_integer('highway_size',    325,            'number of Layer in highway')

# optimization
flags.DEFINE_integer('batch_size',       150,  'number of sequences to train on in parallel')
flags.DEFINE_integer('num_gpus',          7,     'number of GPU for training')
flags.DEFINE_integer('num_threads',       4,       'number of threads')

FLAGS = flags.FLAGS


def main(_):

    test_loader = TextLoader('test', FLAGS.batch_size, FLAGS.num_threads, FLAGS.num_gpus, None)
    print('initialized test dataset reader')

    num_sequences = np.array([3164836, 3126328, 3194929, 3125623])
    num_sequences = num_sequences//(FLAGS.batch_size * FLAGS.num_gpus)

    with tf.device('/cpu:0'):
        test_model_loss = []
        ''' build inference graph '''
        with tf.variable_scope("Model"):
            for gpu_id in range(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % gpu_id):
                    with tf.name_scope('tower_%d' % gpu_id):
                        batch_queue_data = test_loader.batch_data
                        test_model = inference_graph(
                            batch_queue_data[1],
                            batch_queue_data[0],
                            char_vocab_size=6867,
                            char_embed_size=FLAGS.char_embed_size,
                            batch_size=FLAGS.batch_size,
                            num_rnn_layers=FLAGS.rnn_layers,
                            num_highway_layer=FLAGS.num_highway_layer,
                            rnn_size=FLAGS.rnn_size,
                            highway_size=FLAGS.highway_size,
                            dropout=0.0)
                        test_model.update(
                            loss_graph(test_model.logits, batch_queue_data[2],
                                       batch_queue_data[0],
                                       FLAGS.batch_size))
                        tf.get_variable_scope().reuse_variables()
                        test_model_loss.append(test_model.loss)
            batch_test_loss = tf.reduce_mean(test_model_loss)

        sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True))

        ''' training starts here '''
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        num_model = 0
        if ckpt:
            check_point = 'model.ckpt-479135'
            saver.restore(sess, os.path.join(FLAGS.train_dir, check_point))
            global_step = check_point.split('/')[-1].split('-')[-1]
            print ("load model from ", global_step)
            num_batches_test = int(num_sequences.sum())
            all_count = 0
            count = 0
            iter = 0
            test_loss = 0
            loss = 0
            for _ in range(num_batches_test):
                batch_loss = sess.run(batch_test_loss)
                test_loss += batch_loss
                loss += batch_loss
                count += 1
                all_count += 1
                if count == num_sequences[iter]:
                    test_loss /= count
                    print("[%d]: %d: test loss = %.8f, perplexity = %.8f" % (
                        num_model, iter, test_loss, np.exp(test_loss)))
                    iter += 1
                    count = 0
                    test_loss = 0
                if all_count % 500 == 0:
                    print("[%d]: test loss = %.8f, perplexity = %.8f" % (
                        all_count, loss/all_count, np.exp(loss/all_count)))
            loss /= num_batches_test
            print('avg_test_loss/perplexity = %6.8f/%6.7f' % (loss, np.exp(loss)))


if __name__ == "__main__":
    tf.app.run()