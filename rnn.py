# -*- coding: utf-8 -*- 
# @Time : 2018/9/11 13:48 
# @Author : Allen 
# @Site :  构建rnn
import tensorflow as tf


def get_weight(shape):
    return tf.Variable(tf.random_normal(shape))


def get_bias(shape):
    return tf.Variable(tf.random_normal(shape))


def rnn_graph(input_x, dropout_keep_prob, hidden_dim, batch_size, word_len):
    with tf.variable_scope('embedding'):
        embedding = tf.get_variable('embedding', [word_len, hidden_dim])
        lstm_inputs = tf.nn.embedding_lookup(embedding, input_x)
    with tf.variable_scope('BasicLSEMCell'):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
        drop = tf.nn.rnn_cell.DropoutWrapper(lstm, dropout_keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([drop] * 2)
        initial_state = cell.zero_state(batch_size, tf.float32)
        lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, lstm_inputs, initial_state)
        seq_output = tf.concat(lstm_outputs, 1)
    with tf.variable_scope('output'):
        x = tf.reshape(seq_output, [-1, hidden_dim])
        logits = tf.matmul(x, get_weight([hidden_dim, word_len])) + get_bias([word_len])
        prediction = tf.nn.softmax(logits, name='perdictions')
        return logits, prediction, initial_state, final_state


def loss_graph(word_len, target, logits):
    y_one_hot = tf.one_hot(target, word_len)
    y_reshaped = tf.reshape(y_one_hot, [-1, word_len])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, target=y_reshaped))
    return loss


def optimizer_graph(loss, learning_rate):
    grad_clip = 5
    # 使用clipping gradients
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    return optimizer
