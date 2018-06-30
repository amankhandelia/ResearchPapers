import numpy as np
import tensorflow as tf
import pickle as pkl
import os

'''
# TODO
Replace tf.gather with tf.embedding_lookup
Instead of creating three dictionaries create only one for lookup, which we will have embedding for both path as well as value
Then merge it only along the axis where the value where given

'''

def get_dict_lengths(path_to_dictionaries):
    path_vocab = pkl.load(open(os.path.join(path_to_dictionaries, 'path_vocab.pkl'), 'rb'))
    value_vocab = pkl.load(open(os.path.join(path_to_dictionaries, 'value_vocab.pkl'), 'rb'))
    tag_vocab = pkl.load(open(os.path.join(path_to_dictionaries, 'tag_vocab.pkl'), 'rb'))
    return (len(value_vocab), len(path_vocab), len(tag_vocab))

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
    # y['labels'] = tf.squeeze(y['labels'])
    return x, y

path_to_dic = '/home/weave/Documents/vocab_dictionaries'
d, max_seq_len = 128, 100
value_vocab_len, path_vocab_len, tags_vocab_len = get_dict_lengths(path_to_dic)
batch_size, epochs = 100, 10000
key_names = ['length', 'sequence', 'labels']

def loss(Y, prediction):
    loss_total = tf.reduce_sum(-1.0*tf.multiply(tf.cast(Y, dtype=tf.float64), tf.log(prediction)), axis=0)
    loss_ = tf.reduce_mean(loss_total)
    return loss_

def model_fn(X, Y):
    graph = tf.get_default_graph()
    with graph.as_default():
        with tf.variable_scope('code2vec', reuse=tf.AUTO_REUSE):
        #     d = tf.placeholder(name='dimension_of_everything', shape=(), dtype=tf.int32)
        #     X = tf.placeholder(name='path_context', shape=(None, None, 3), dtype=tf.int32)
        #     Y = tf.placeholder(name='true_distribution', shape=(None, tags_vocab_len), dtype=tf.float32)
            path_vocab = tf.get_variable(name='path_vocab', shape=(path_vocab_len, d), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            value_vocab = tf.get_variable(name='value_vocab', shape=(value_vocab_len, d), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            tags_vocab = tf.get_variable(name='tags_vocab', shape=(tags_vocab_len, d), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            W = tf.get_variable(name='context_vector_transformation_matrix', shape=(d, 3*d), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            attention_weight_vector = tf.get_variable('pay_attention_to_context', shape=(d, 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            x_part_one = tf.squeeze(X[:,:, 0])
            x_part_two = tf.squeeze(X[:,:, 1])
            x_part_three = tf.squeeze(X[:,:,2])
            x_part_one_embedding = tf.gather(value_vocab, x_part_one, name='gather_value_first_vocab')
            x_part_three_embedding = tf.gather(value_vocab, x_part_three, name='gather_value_second_vocab')
            x_part_two_embedding = tf.gather(path_vocab, x_part_two, name='gather_path_vocab')
            context_vectors = tf.concat([x_part_one_embedding, x_part_two_embedding, x_part_three_embedding], axis=2)
            context_vectors_shape = tf.shape(context_vectors)
            context_vectors = tf.reshape(context_vectors, shape=(context_vectors_shape[0], context_vectors_shape[1], context_vectors_shape[2]))
        #     context_vectors = get_embeddings(X)
        #     context_vectors = tf.cast(X, dtype=tf.float32)
            #dimension W: d(i) X 3*d(j), context_vector: batch_size(l), None(k), 3*d(j), context_vectors_transformed: batch_size(l), None(k), d(i)
            #One thing to take note is that here the assumption is einsum can work with variable-length-sequence vectors
            context_vectors_transformed = tf.einsum('ij,lkj->lki', W, context_vectors)
            #dimension attention_weight_vector: d(i) X 1(m)
            attention_weight_alpha_inner_product = tf.einsum('lki,im->lk', context_vectors_transformed, attention_weight_vector)
            attention_weight_alpha = tf.nn.softmax(attention_weight_alpha_inner_product, axis=0)
            code_vectors = tf.einsum('lki,lk->li', context_vectors_transformed, attention_weight_alpha)
            prediction_inner_product = tf.einsum('li,pi->lp', code_vectors, tags_vocab)
            prediction = tf.nn.softmax(prediction_inner_product, axis=0)
            _loss_ = loss(Y, prediction)

    return (_loss_, prediction, x_part_one_embedding, x_part_two_embedding, x_part_three_embedding)

graph = tf.get_default_graph()
with graph.as_default():
    with tf.variable_scope('code2vec', reuse=tf.AUTO_REUSE):
        path_vocab = tf.get_variable(name='path_vocab', shape=(path_vocab_len, d),
                                     initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        value_vocab = tf.get_variable(name='value_vocab', shape=(value_vocab_len, d),
                                      initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        tags_vocab = tf.get_variable(name='tags_vocab', shape=(tags_vocab_len, d),
                                     initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        W = tf.get_variable(name='context_vector_transformation_matrix', shape=(d, 3 * d),
                            initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        attention_weight_vector = tf.get_variable('pay_attention_to_context', shape=(d, 1),
                                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        # Let's write down the data pipeline for the
        path_to_tfrecord  = '/home/weave/Documents/tfrecord_files/*.tfrecord'
        file_list = tf.data.Dataset.list_files(path_to_tfrecord)
        dataset = tf.data.TFRecordDataset(file_list).repeat()
        dataset = dataset.map(parse_sequence).map(inflate).padded_batch(batch_size, padded_shapes=({'length':1}, {'sequence':tf.TensorShape([None, 3]), 'labels':1})).map(deflate)
        iterator = dataset.make_one_shot_iterator()
        _, data = iterator.get_next()
        X, Y = data['sequence'], data['labels']
        Y = tf.squeeze(tf.one_hot(Y, tags_vocab_len, axis=-1))
        model_loss, prediction, x_part_one_embedding, x_part_two_embedding, x_part_three_embedding = model_fn(X, Y)
        accuracy = tf.metrics.accuracy(Y, prediction)
        optimizer = tf.train.AdamOptimizer().minimize(model_loss)
        x_part_one_embedding, x_part_two_embedding, x_part_three_embedding = None, None, None
    with tf.Session() as sess:
        options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        init = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init)
        sess.run(init_l)
        for epoch in range(epochs):
            # _, data = sess.run(dataset)
            # data['labels'] = np.eye(tags_vocab_len)[data['labels']]
            # print(data['labels'])
            _, _loss_, _accuracy_ = sess.run([optimizer, model_loss, accuracy], options = options)
            print('Epoch: ', epoch, 'Loss: ', _loss_, 'Accuracy: ', _accuracy_)
    #         _shape_ = sess.run(tf.shape(x_part_one_embedding), feed_dict={X:data['sequence'], Y:data['labels']})
    #         print(_shape_)