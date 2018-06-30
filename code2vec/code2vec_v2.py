from operator import concat

import numpy as np
import tensorflow as tf
import pickle as pkl
import os

def get_dict_lengths(path_to_dictionaries):
    value_path_vocab = pkl.load(open(os.path.join(path_to_dictionaries, 'path_and_value_vocab.pkl'), 'rb'))
    path_vocab = pkl.load(open(os.path.join(path_to_dictionaries, 'path_vocab.pkl'), 'rb'))
    value_vocab = pkl.load(open(os.path.join(path_to_dictionaries, 'value_vocab.pkl'), 'rb'))
    tag_vocab = pkl.load(open(os.path.join(path_to_dictionaries, 'tag_vocab.pkl'), 'rb'))
    return len(value_vocab), len(path_vocab), len(tag_vocab), len(value_path_vocab)


path_to_tfrecord  = '/home/weave/Documents/code2vec_data/github_dataset/tfrecord_files/train/*.tfrecord'
path_to_dic = '/home/weave/Documents/code2vec_data/github_dataset/dictionaries/'
path_to_model = '/home/weave/Documents/code2vec_data/github_dataset/checkpoints'
model_name = 'code2vec_150k'
d, max_seq_len = 8, 100
value_vocab_len, path_vocab_len, tags_vocab_len, value_path_vocab_len = 11575702+1, 351094+1, 3366248+1, 11926796+1 #get_dict_lengths(path_to_dic)
batch_size, epochs, learning_rate = 100, 100000, 10e-3
saver_gap = 10000
key_names = ['length', 'sequence', 'labels']



def parse_sequence(datapoint_raw):
    context_features = {
        "length":tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_feature = {
        "sequence": tf.FixedLenSequenceFeature([3], dtype=tf.int64),
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }
    datapoint = tf.parse_single_sequence_example(datapoint_raw, context_features=context_features, sequence_features=sequence_feature)
    return datapoint

def inflate(x, y):
    x['length'] = tf.expand_dims(tf.convert_to_tensor(x['length']),0)
    return x, y
def deflate(x, y):
    x['length'] = tf.squeeze(x['length'])
    return x, y


def loss(Y, prediction):
    loss_total = tf.reduce_sum(-1.0*tf.multiply(tf.cast(Y, dtype=tf.float64), tf.log(prediction)), axis=0)
    loss_ = tf.reduce_mean(loss_total)
    return loss_

def model_fn(X, Y):
    """
    Implementation Notes:
        dimension W: d(i) X 3*d(j), context_vector: batch_size(l), None(k), 3*d(j), context_vectors_transformed: batch_size(l), None(k), d(i)
        One thing to take note is that here the assumption is einsum can work with variable-length-sequence vectors
        dimension attention_weight_vector: d(i) X 1(m)
    :param X: Sequences, a set of path_context as defined in the code2vec paper
    :param Y: Labels, method name using which we will
    :return: Loss, Predictions
    """
    graph = tf.get_default_graph()
    with graph.as_default():
        with tf.variable_scope('code2vec', reuse=tf.AUTO_REUSE):
            value_and_path_vocab = tf.get_variable(name="embedding_dict_for_path_and_vocab", shape=(value_path_vocab_len, d), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            tags_vocab = tf.get_variable(name='tags_vocab', shape=(tags_vocab_len, d), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            W = tf.get_variable(name='context_vector_transformation_matrix', shape=(d, 3*d), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            attention_weight_vector = tf.get_variable('pay_attention_to_context', shape=(d, 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            context_vectors = tf.nn.embedding_lookup(value_and_path_vocab, X)
            context_vectors = tf.reshape(context_vectors, [tf.shape(context_vectors)[0], tf.shape(context_vectors)[1], tf.shape(context_vectors)[2] * tf.shape(context_vectors)[3]])
            context_vectors_transformed = tf.einsum('ij,lkj->lki', W, context_vectors)
            attention_weight_alpha_inner_product = tf.einsum('lki,im->lk', context_vectors_transformed, attention_weight_vector)
            attention_weight_alpha = tf.nn.softmax(attention_weight_alpha_inner_product, axis=0)
            code_vectors = tf.einsum('lki,lk->li', context_vectors_transformed, attention_weight_alpha)
            prediction_inner_product = tf.einsum('li,pi->lp', code_vectors, tags_vocab)
            prediction = tf.nn.softmax(prediction_inner_product, axis=0)
            model_fn_loss = loss(Y, prediction)
    return (model_fn_loss, prediction)

graph = tf.get_default_graph()
with graph.as_default():
    with tf.variable_scope('code2vec', reuse=tf.AUTO_REUSE):
        value_and_path_vocab = tf.get_variable(name="embedding_dict_for_path_and_vocab",
                                               shape=(value_path_vocab_len, d),
                                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        tags_vocab = tf.get_variable(name='tags_vocab', shape=(tags_vocab_len, d),
                                     initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        W = tf.get_variable(name='context_vector_transformation_matrix', shape=(d, 3 * d),
                            initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        attention_weight_vector = tf.get_variable('pay_attention_to_context', shape=(d, 1),
                                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        # Let's write down the data pipeline for reading the data
        file_list = tf.data.Dataset.list_files(path_to_tfrecord)
        dataset = tf.data.TFRecordDataset(file_list).repeat()
        dataset = dataset.map(parse_sequence).map(inflate).padded_batch(batch_size, padded_shapes=({'length':1}, {'sequence':tf.TensorShape([None, 3]), 'labels':1})).map(deflate)
        iterator = dataset.make_one_shot_iterator()
        _, data = iterator.get_next()
        X, Y = data['sequence'], data['labels']
        X_shape = tf.shape(X)
        Y = tf.squeeze(tf.one_hot(Y, tags_vocab_len, axis=-1))
        model_loss, predictions = model_fn(X, Y)
        accuracy = tf.metrics.accuracy(Y, predictions)
        optimizer = tf.train.AdamOptimizer().minimize(model_loss)
    with tf.Session() as sess:
        saver = tf.train.Saver([value_and_path_vocab, tags_vocab, W, attention_weight_vector])
        # saver = tf.train.Saver()
        options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        init = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init)
        sess.run(init_l)
        for epoch in range(epochs):
            _, _loss_, _accuracy_, input_shape = sess.run([optimizer, model_loss, accuracy, X_shape], options = options)
            print('Epoch: ', epoch, 'Loss: ', _loss_, 'Accuracy: ', _accuracy_, "input_shape: ", input_shape)
            if epoch % saver_gap == 0:
                saver.save(sess, os.path.join(path_to_model, model_name), global_step=epoch)