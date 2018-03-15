import codecs
import os
import collections
from six.moves import cPickle
import numpy as np
import tensorflow as tf


def make_example(sequence, labels):
    length_features = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=[len(sequence)]))
    ]
    input_features = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=[token]))
        for token in sequence]
    label_features = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        for label in labels]
    feature_list = {
        'length': tf.train.FeatureList(feature=length_features),
        'inputs': tf.train.FeatureList(feature=input_features),
        'labels': tf.train.FeatureList(feature=label_features)
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return tf.train.SequenceExample(feature_lists=feature_lists)


class TextLoader():
    def __init__(self, signal, batch_size, num_threads, num_gpus, max_epochs):

        if signal == 'train':
            str_file = 'abcdefghijklm'
            tfrecord = []
            for s in str_file:
                file = os.path.join('data', 'train_a' + s + '_tfrecord')
                tfrecord.append(file)
        if signal == 'valid':
            file = os.path.join('data', 'valid_tfrecord')
            tfrecord = [file]
        if signal == 'test':
            str_file = 'abcd'
            tfrecord = []
            for s in str_file:
                file = os.path.join('data', 'diffa' + s + '_tfrecord')
                tfrecord.append(file)

        self.batch_data = self.read_process(tfrecord, batch_size, num_threads, num_gpus, max_epochs)

    def read_process(self, filename, batch_size, num_threads,  num_gpus, max_epochs):
        reader = tf.TFRecordReader()
        file_queue = tf.train.string_input_producer(filename, shuffle=False, num_epochs=max_epochs)
        key, serialized_example = reader.read(file_queue)
        sequence_features = {
            'length': tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "inputs": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }
        # Parse the example (returns a dictionary of tensors)
        _, sequence_parsed = tf.parse_single_sequence_example(
            serialized_example,
            sequence_features=sequence_features
        )
        input_tensors = [sequence_parsed['length'], sequence_parsed['inputs'], sequence_parsed['labels']]
        return tf.train.batch(
            input_tensors,
            batch_size=batch_size,
            capacity=10 + num_gpus * batch_size,
            num_threads=num_threads,
            dynamic_pad=True,
            allow_smaller_final_batch=False
        )
