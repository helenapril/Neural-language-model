from __future__ import print_function
from __future__ import division

import tensorflow as tf


class adict(dict):
    ''' Attribute dictionary - a convenience data structure, similar to SimpleNamespace in python 3.3
        One can use attributes to read/write dictionary content.
    '''

    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


def linear(input_, output_size, scope=None):
    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """
    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))
            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)
            output = t * g + (1. - t) * input_
            input_ = output

    return input_


def inference_graph(input_,
                    length,
                    char_vocab_size,
                    char_embed_size=15,
                    batch_size=20,
                    num_rnn_layers=2,
                    num_highway_layer=2,
                    rnn_size=650,
                    highway_size=1025,
                    dropout=0.0):
    num_unroll_steps = tf.shape(input_)[1]
    input_ = tf.reshape(input_, [batch_size, num_unroll_steps], name='input_')
    length = tf.reshape(length, [batch_size])
    ''' First, embed characters '''
    with tf.variable_scope('Embedding'):
        char_embedding = tf.get_variable('char_embedding', [char_vocab_size, char_embed_size])
        input_embedded = tf.nn.embedding_lookup(char_embedding, input_)

    ''' Finally, do LSTM '''
    with tf.variable_scope('LSTM'):
        def create_rnn_cell():
            #cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=1.0)
            cell = tf.contrib.rnn.BasicRNNCell(rnn_size)
            if dropout > 0.0:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1. - dropout)
            return cell

        if num_rnn_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([create_rnn_cell() for _ in range(num_rnn_layers)], state_is_tuple=True)
        else:
            cell = create_rnn_cell()

        initial_rnn_state = cell.zero_state(batch_size, dtype=tf.float32)
        outputs, final_rnn_state = tf.nn.dynamic_rnn(cell, input_embedded, sequence_length=length,
                                                     initial_state=initial_rnn_state, dtype=tf.float32)
        outputs = tf.reshape(outputs, [batch_size, num_unroll_steps, rnn_size])

        with tf.variable_scope('PredictEmbedding') as scope:
            outputs = tf.reshape(outputs, [-1, rnn_size])
            logits = linear(outputs, char_vocab_size)

    return adict(
        initial_rnn_state=initial_rnn_state,
        logits=logits
    )


def loss_graph(logits, targets, length, batch_size):
    with tf.variable_scope('Loss'):
        length = tf.reshape(length, [batch_size])
        num_unroll_steps = tf.shape(targets)[-1]
        targets_flat = tf.reshape(targets, [batch_size * num_unroll_steps])
        mask = tf.sign(tf.to_float(targets_flat))
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets_flat)
        masked_loss = mask * loss
        masked_loss = tf.reshape(masked_loss, [batch_size, num_unroll_steps])
        length = tf.cast(length, tf.float32)
        mean_loss_by_example = tf.reduce_sum(masked_loss, 1)/length
        mean_loss = tf.reduce_mean(mean_loss_by_example)
        real_loss = tf.reduce_sum(masked_loss)
        length = tf.reduce_sum(length)

    return adict(
        loss=mean_loss,
        real_loss=real_loss,
        length=length
    )




