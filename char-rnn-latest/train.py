from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf

from record_reader import TextLoader
from model import inference_graph, loss_graph

flags = tf.flags

# data
flags.DEFINE_string('data_dir',    'data',   'data directory')
flags.DEFINE_string('train_dir',   'cv',     'training directory')
flags.DEFINE_string('load_model',   None,    '(optional) filename of the model to load')

# model params
flags.DEFINE_integer('rnn_size',        725,            'size of LSTM internal state')
flags.DEFINE_integer('char_embed_size', 325,            'dimensionality of character embeddings')
flags.DEFINE_integer('rnn_layers',      1,              'number of layers in the LSTM')
flags.DEFINE_integer('num_highway_layer', 0,            'number of highway layer')
flags.DEFINE_float  ('dropout',         0.5,            'dropout. 0 = no dropout')
flags.DEFINE_integer('highway_size',    325,            'number of Layer in highway')

# optimization
flags.DEFINE_float  ('learning_rate_decay', 0.5,  'learning rate decay')
flags.DEFINE_float  ('learning_rate',       0.1,  'starting learning rate')
flags.DEFINE_float  ('param_init',          0.05, 'initialize parameters at')
flags.DEFINE_integer('batch_size',          150,   'number of sequences to train on in parallel')
flags.DEFINE_integer('max_epochs',          7,      'number of full passes through the training data')
flags.DEFINE_float  ('max_grad_norm',       5.0,  'normalize gradients at')
flags.DEFINE_integer('decay_epochs',        2,   'number of full passes through the training data')
flags.DEFINE_integer('num_gpus',            7,     'number of GPU for training')
flags.DEFINE_integer('min_capacity',        10,     'num of remaining data')
flags.DEFINE_integer('num_threads',         4,       'number of threads')

# bookkeeping
flags.DEFINE_integer('print_every',    100,    'how often to print current loss')
flags.DEFINE_string ('EOS',            '+',  '<EOS> symbol')

FLAGS = flags.FLAGS


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = [g for g, _ in grad_and_vars]
        grad = tf.stack(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def main(_):

    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)
        print('Created training directory', FLAGS.train_dir)

    train_loader = TextLoader('train', FLAGS.batch_size, FLAGS.num_threads, FLAGS.num_gpus, None)
    valid_loader = TextLoader('valid', FLAGS.batch_size, FLAGS.num_threads, FLAGS.num_gpus, None)
    print('initialized readers')

    num_sequences = np.array([26025455, 25855894, 25935222, 25884830, 25880158, 25987568,
                              25870758, 25714083, 25598004, 25653803, 25570567, 25532410, 25583980])

    '''define graph'''
    with tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        num_batches_train = int(num_sequences.sum() / (FLAGS.batch_size * FLAGS.num_gpus))
        print ('num_batches_train: %d' % num_batches_train)
        decay_steps = num_batches_train * FLAGS.decay_epochs
        lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                        global_step,
                                        decay_steps,
                                        FLAGS.learning_rate_decay,
                                        staircase=True)
        opt = tf.train.GradientDescentOptimizer(lr)

        tower_models = []
        initializer = tf.random_uniform_initializer(-FLAGS.param_init, FLAGS.param_init)
        with tf.variable_scope('Model', initializer=initializer):
            for gpu_id in range(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % gpu_id):
                    with tf.name_scope('tower_%d' % gpu_id):
                        batch_queue_data = train_loader.batch_data
                        train_model = inference_graph(
                            batch_queue_data[1],
                            batch_queue_data[0],
                            char_vocab_size=6867,
                            char_embed_size=FLAGS.char_embed_size,
                            batch_size=FLAGS.batch_size,
                            num_rnn_layers=FLAGS.rnn_layers,
                            num_highway_layer=FLAGS.num_highway_layer,
                            rnn_size=FLAGS.rnn_size,
                            highway_size=FLAGS.highway_size,
                            dropout=FLAGS.dropout)
                        train_model.update(
                            loss_graph(train_model.logits, batch_queue_data[2],
                                       batch_queue_data[0],
                                       FLAGS.batch_size))
                        tf.get_variable_scope().reuse_variables()
                        gradient, tvar = zip(*opt.compute_gradients(train_model.loss))
                        gradient, _ = tf.clip_by_global_norm(gradient, FLAGS.max_grad_norm)
                        grads = zip(gradient, tvar)
                        tower_models.append([train_model.initial_rnn_state, train_model.loss, grads])

            initial_rnn_state, tower_losses, tower_grads = zip(*tower_models)
            train_batch__loss = tf.reduce_mean(tower_losses)
            apply_gradient_op = opt.apply_gradients(average_gradients(tower_grads), global_step=global_step)


        '''model for validation model'''
        ''' build graph for validation and testing (shares parameters with the training graph!) '''
        valid_model_loss = []
        with tf.variable_scope("Model", reuse=True):
            for gpu_id in range(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % gpu_id):
                    with tf.name_scope('tower_%d' % gpu_id):
                        batch_queue_data = valid_loader.batch_data
                        valid_model = inference_graph(
                            batch_queue_data[1],
                            batch_queue_data[0],
                            char_vocab_size=6867,
                            char_embed_size=FLAGS.char_embed_size,
                            batch_size=FLAGS.batch_size,
                            num_rnn_layers=FLAGS.rnn_layers,
                            num_highway_layer=FLAGS.num_highway_layer,
                            rnn_size=FLAGS.rnn_size,
                            highway_size=FLAGS.highway_size,
                            dropout=FLAGS.dropout)
                        valid_model.update(
                            loss_graph(valid_model.logits, batch_queue_data[2],
                                       batch_queue_data[0],
                                       FLAGS.batch_size))
                        tf.get_variable_scope().reuse_variables()
                        valid_model_loss.append(valid_model.loss)
            valid_batch_loss = tf.reduce_mean(valid_model_loss)

        '''start training'''
        saver = tf.train.Saver(max_to_keep=50)

        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True))
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        sess.run(initial_rnn_state)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        train_start_time = 0

        for epoch in range(FLAGS.max_epochs):
            epoch_start_time = time.time()
            avg_train_loss = 0.0
            count = 0
            for step in range(num_batches_train):
                count += 1
                loss, _, g_step = sess.run([train_batch__loss, apply_gradient_op, global_step])
                avg_train_loss += loss
                if count % FLAGS.print_every == 0:
                    print('%d: [%d/%d], train_loss/perplexity = %6.8f/%6.7f' %
                          (epoch, count, num_batches_train, loss, np.exp(loss)))
                if g_step % 500 == 0:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=global_step)
                    print('Saved model', checkpoint_path)

                    '''start validation set'''
                    if count % 1000 == 0:
                        num_batches_valid = int(3800000 / (FLAGS.batch_size * FLAGS.num_gpus))
                        avg_valid_loss = 0
                        cnt = 0
                        g_step = sess.run(global_step)
                        for step in range(num_batches_valid):
                            loss = sess.run(valid_batch_loss)
                            cnt += 1
                            avg_valid_loss += loss
                            if cnt % FLAGS.print_every == 0:
                                print('[%d/%d]: [%d/%d], valid_loss/perplexity = %6.8f/%6.7f' %
                                      (epoch, g_step, cnt, num_batches_valid, loss, np.exp(loss)))

                        avg_valid_loss /= cnt
                        print("at the end of epoch: %d, at the step: %d" % (epoch, g_step))
                        print("valid loss = %6.8f, perplexity = %6.8f" % (avg_valid_loss, np.exp(avg_valid_loss)))

            avg_train_loss /= count
            epoch_time = time.time()-epoch_start_time
            train_start_time += epoch_time
            print('Epoch training time:', epoch_time)
            print("at the end of epoch:", epoch)
            print("train loss = %6.8f, perplexity = %6.8f" % (avg_train_loss, np.exp(avg_train_loss)))

            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)
            print('Saved model', checkpoint_path)

            num_batches_valid = int(3800000 / (FLAGS.batch_size * FLAGS.num_gpus))
            avg_valid_loss = 0
            cnt = 0
            for step in range(num_batches_valid):
                loss = sess.run(valid_batch_loss)
                cnt += 1
                avg_valid_loss += loss
                if cnt % FLAGS.print_every == 0:
                    g_step = sess.run(global_step)
                    print('[%d/%d]: [%d/%d], valid_loss/perplexity = %6.8f/%6.7f' %
                          (epoch, g_step, cnt, num_batches_valid, loss, np.exp(loss)))

            avg_valid_loss /= cnt
            print("at the end of epoch:", epoch)
            print("valid loss = %6.8f, perplexity = %6.8f" % (avg_valid_loss, np.exp(avg_valid_loss)))

        print('All training time:', train_start_time)


if __name__ == "__main__":
    tf.app.run()