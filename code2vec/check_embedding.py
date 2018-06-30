import tensorflow as tf
import numpy as np


epochs = 10
vocab_size = 5
embedding_size = 3
embedding_dict = tf.get_variable("embedding_dict", [vocab_size, embedding_size])
embedded_output = tf.nn.embedding_lookup(embedding_dict, [[1,2],[2,1]])
# logits = tf.layers.dense(inputs=embedded_output, units=2)
# output = tf.nn.softmax(logits)
# Y = np.array([[0,1],[1,0]])
# loss = tf.losses.mean_squared_error(Y, output)
# optimizer = tf.train.AdamOptimizer().minimize(loss)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    prev_matrix = np.random.randn(vocab_size, embedding_size)
    for i in range(epochs):
        __embedded_output__, _embedding_dict_ = sess.run([embedded_output, embedding_dict])
        print(__embedded_output__)
        # print(_embedding_dict_ == prev_matrix)
        # prev_matrix = _embedding_dict_
        # print(np.unstack(prev_matrix, 1))
