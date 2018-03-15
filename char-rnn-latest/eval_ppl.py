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
flags.DEFINE_string('train_dir',   'double',     'training directory')

# model params
flags.DEFINE_integer('rnn_size',        725,        'size of LSTM internal state')
flags.DEFINE_integer('char_embed_size', 325,        'dimensionality of character embeddings')
flags.DEFINE_integer('rnn_layers',      2,          'number of layers in the LSTM')
flags.DEFINE_integer('num_highway_layer', 0,            'number of highway layer')
flags.DEFINE_integer('highway_size',    325,            'number of Layer in highway')

# optimization
flags.DEFINE_integer('batch_size',        150,  'number of sequences to train on in parallel')
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
        test_model_length = []
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
                        test_model_loss.append(test_model.real_loss)
                        test_model_length.append(test_model.length)
            batch_test_loss = tf.reduce_sum(test_model_loss)
            batch_test_length = tf.reduce_sum(test_model_length)

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
            check_point = 'model.ckpt-66000'
            saver.restore(sess, os.path.join(FLAGS.train_dir, check_point))
            global_step = check_point.split('/')[-1].split('-')[-1]
            print ("load model from ", global_step)
            num_batches_test = int(num_sequences.sum())
            all_count = 0
            all_length = 0
            count = 0
            iter = 0
            test_loss = 0
            test_length = 0
            loss = 0
            for _ in range(num_batches_test):
                batch_loss, batch_length = sess.run([batch_test_loss, batch_test_length])
                test_loss += batch_loss
                test_length += batch_length
                loss += batch_loss
                count += 1
                all_count += 1
                all_length += batch_length
                if count == num_sequences[iter]:
                    test_loss /= (test_length + count*(FLAGS.batch_size * FLAGS.num_gpus))
                    print("[%d]: %d: test loss = %.8f, perplexity = %.8f" % (
                        num_model, iter, test_loss, np.exp(test_loss)))
                    iter += 1
                    count = 0
                    test_loss = 0
                    test_length = 0
                if all_count % 500 == 0:
                    temp = loss/(all_length + all_count*FLAGS.batch_size * FLAGS.num_gpus)
                    print("[%d]: test loss = %.8f, perplexity = %.8f" % (
                        all_count, temp, np.exp(temp)))
            loss /= (all_length + all_count*FLAGS.batch_size * FLAGS.num_gpus)
            print('avg_test_loss/perplexity = %6.8f/%6.7f' % (loss, np.exp(loss)))


if __name__ == "__main__":
    tf.app.run()