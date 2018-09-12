# -*- coding: utf-8 -*- 
# @Time : 2018/9/11 11:44 
# @Author : Allen 
# @Site :  训练rnn模型
import os

import tensorflow as tf
from rnn import rnn_graph, loss_graph, optimizer_graph
from data_helper import Poem
from datetime import datetime


def train(batch_size, hidden_dim, learning_rate, epochs, file_path, dropout_keep_prob, model_path):
    poem = Poem(file_path, batch_size)
    word_len = len(poem.word_to_int)
    input_x = tf.placeholder(tf.int32, [batch_size, None], name='inputs')
    target = tf.placeholder(tf.int32, [batch_size, None], name='target')
    keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
    logits, _, initial_state, final_state = rnn_graph(input_x, dropout_keep_prob, hidden_dim, batch_size, word_len)
    loss = loss_graph(word_len, target, logits)
    optimizer = optimizer_graph(loss, learning_rate)

    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        new_state = sess.run(initial_state)
        step = 0
        for i in range(epochs):
            batches = poem.batch_iter()
            for batch_x, batch_y in batches:
                feed = {
                    input_x: batch_x,
                    target: batch_y,
                    initial_state: new_state,
                    keep_prob: dropout_keep_prob,
                }
                _loss, _, new_state = sess.run([loss, optimizer, final_state], feed_dict=feed)
                print(datetime.now().strftime('%c'), 'epoch:', i, 'step:', step + 1, 'loss:', _loss)
                step += 1
        saver.save(sess, os.path.join(model_path, 'model.ckpt'))
