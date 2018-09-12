# -*- coding: utf-8 -*- 
# @Time : 2018/9/11 11:45 
# @Author : Allen 
# @Site :  数据处理文件
import numpy as np


class Poem:
    def __init__(self, path, batch_size):
        self.file_path = path
        self.poem_list = self.get_poem()
        self.poem_vectors, self.word_to_int, self.int_to_word = self.get_poem_vectors()
        self.batch_size = batch_size
        self.chunk_size = len(self.poem_vectors) // self.batch_size

    def get_poem(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            poem_list = ([line.strip().split(':')[-1] for line in f.readlines()])
        return poem_list

    def get_poem_vectors(self):
        words = sorted(set(''.join(self.poem_list) + ' '))
        int_to_word = {i: word for i, word in enumerate(words)}
        word_to_int = {v: k for k, v in int_to_word.items()}
        to_int = lambda word: word_to_int.get(word)
        poem_vectors = [list(map(to_int, poem)) for poem in self.poem_list]
        return poem_vectors, word_to_int, int_to_word

    def batch_iter(self):
        start = 0
        end = self.batch_size
        for _ in range(self.chunk_size):
            batches = self.poem_vectors[start:end]
            x_batch = np.full((self.batch_size, max(map(len, batches))), self.word_to_int[' '], np.int32)
            for row in range(self.batch_size):
                x_batch[row, :len(batches[row])] = batches[row]
            y_batch = np.copy(x_batch)
            y_batch[:, :-1], y_batch[:, -1] = x_batch[:, 1:], x_batch[:, 0]
            yield x_batch, y_batch
            start += self.batch_size
            end += self.batch_size


if __name__ == '__main__':
    poem = Poem(r'D:\workspace\poemRNN\data\poems.txt', 128)
    data = poem.batch_iter()
    for x, y in data:
        print(x)
        print(y)
