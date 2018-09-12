# -*- coding: utf-8 -*- 
# @Time : 2018/9/11 11:44 
# @Author : Allen 
# @Site :  主文件
import configparser
import data_helper
import os
from train import train


def exists_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


cf = configparser.ConfigParser()
cf.read('configuration.ini')
learning_rate = float(cf.get('para', 'learning_rate'))
epochs = int(cf.get('para', 'epochs'))
batch_size = int(cf.get('para', 'batch_size'))
data_path = exists_dir(os.path.join(os.getcwd(), cf.get('para', 'data_path')))
model_path = exists_dir(os.path.join(os.getcwd(), cf.get('para', 'model_path')))
summaries_path = exists_dir(os.path.join(os.getcwd(), cf.get('para', 'summaries_path')))
hidden_dim = int(cf.get('para', 'hidden_dim'))
file_path = os.path.join(os.getcwd(), cf.get('para', 'file_path'))
dropout_keep_prob = float(cf.get('para', 'dropout_keep_prob'))


def predict():
    pass


def main():
    '''
    判断模型是否存在
    :return:
    '''
    if exists_dir(model_path) and os.path.getsize(model_path) > 0:
        predict()
    else:
        train(batch_size, hidden_dim, learning_rate, epochs, file_path, dropout_keep_prob, model_path)


if __name__ == '__main__':
    main()
