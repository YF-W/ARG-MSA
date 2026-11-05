# -*- coding: utf-8 -*-

from __future__ import division
from torch.utils.data import DataLoader, TensorDataset
from preprocessor.dictionary import Dictionary
from preprocessor.embedding import Embedding
from utils.generic import clean_tensor
import pickle
import torch
import os
import numpy as np


class SIMSReader(object):
    def __init__(self, opt):
        self.embedding_enabled = True
        self.all_feature_names = ['textual', 'visual', 'acoustic']
        self.feature_indexes = [self.all_feature_names.index(f.strip()) for f in opt.features.split(',')]
        self.fp_prefix = os.path.join(opt.pickle_dir_path, 'sims')  # changed here
        self.data_path = "C:/datasets/SIMS/Processed/unaligned_39.pkl"
        print("[DEBUG] data_path = ", self.data_path) 

    def read(self, opt):
        for key, value in opt.__dict__.items():
            if not key == 'feature_indexes':
                self.__setattr__(key, value)
        X_train, X_test, X_dev, y_train, y_test, y_dev = self.load_pickle_data()

        self.datas = {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test},
            'dev': {'X': X_dev, 'y': y_dev}
        }

        self.train_sample_num = len(X_train[0])
        self.output_dim = 1
        self.input_dims = [x.shape[-1] for i, x in enumerate(X_train) if (i in self.feature_indexes)]

        if self.embedding_enabled:
            self.input_dims[0] = self.embedding.embedding_size

        self.opt_callback(opt)

    def opt_callback(self, opt):
        opt.dataset_name = 'sims'
        opt.feature_indexes = self.feature_indexes
        opt.input_dims = self.input_dims
        opt.train_sample_num = self.train_sample_num
        opt.output_dim = self.output_dim
        opt.embedding_enabled = self.embedding_enabled
        opt.max_seq_len = self.max_seq_len
        if 'embedding' in self.__dict__:
            opt.lookup_table = self.embedding.lookup_table
        if 'emotion_dic' in self.__dict__:
            opt.emotion_dic = self.emotion_dic

    def get_max_seq_len(self, features):
        if 'max_seq_len' not in self.__dict__:
            self.max_seq_len = max([len(feature) for feature in features])

    def load_pickle_data(self):
    

        data = pickle.load(open(self.data_path, 'rb'))
        X_train_path = self.fp_prefix + '_train.pkl'
        X_test_path = self.fp_prefix + '_test.pkl'
        X_dev_path = self.fp_prefix + '_valid.pkl'

        # 计算最大序列长度（可选，用于 padding）
        self.get_max_seq_len(np.concatenate([
            data['train']['text'],
            data['test']['text'],
            data['valid']['text']
        ], axis=0))

        # 判断是否需要构建嵌入
        embedding_path = self.fp_prefix + '_embedding.pkl'
        textual_features = np.concatenate([
            data['train']['text'],
            data['test']['text'],
            data['valid']['text']
        ], axis=0)

        # 如果是原始的 token 列表（字符串），才需要构建 embedding
        if textual_features.dtype.type is np.str_ or textual_features.dtype == object:
            if not os.path.exists(embedding_path):
                print("Creating new embeddings!")
                self.dictionary = Dictionary(start_feature_id=0)
                self.dictionary.add('UNK')

                for tokens in textual_features:
                    for token in tokens:
                        if isinstance(token, str):
                            self.dictionary.add(token.lower())
                        else:
                            self.dictionary.add(str(token))

                self.embedding = Embedding(self.dictionary, self.max_seq_len)
                self.embedding.get_embedding(dataset_name='sims', fname=self.wordvec_path)
                pickle.dump(self.embedding, open(embedding_path, 'wb'))
            else:
                print("  - Found cached embeddings")
                self.embedding = pickle.load(open(embedding_path, 'rb'))
        else:
            print("  - Detected pre-embedded features, skipping embedding construction")
            self.embedding = None

            # 返回原始划分的样本
            return data['train']['text'], data['test']['text'], data['valid']['text'], \
                data['train']['labels'], data['test']['labels'], data['valid']['labels']


        def make_or_load_split(split_name):
            split_path = self.fp_prefix + f'_{split_name}.pkl'
            if not os.path.exists(split_path):
                print(f"Creating new {split_name} data!")
                X = [[self.embedding.text_to_sequence(seq) for seq in data[split_name]['text']],
                     data[split_name]['vision'],
                     data[split_name]['audio']]
                y = data[split_name][self.label]
                pickle.dump([*X, y], open(split_path, 'wb'))
            else:
                print(f"  - Found cached {split_name} data")
                loaded = pickle.load(open(split_path, 'rb'))
                X = loaded[:-1]
                y = loaded[-1]
            return X, y

        X_train, y_train = make_or_load_split('train')
        X_test, y_test = make_or_load_split('test')
        X_dev, y_dev = make_or_load_split('valid')

        def tensorize(X):
            return [torch.tensor(x, dtype=torch.int64) if i == 0 else torch.tensor(x, dtype=torch.float32)
                    for i, x in enumerate(X)]

        X_train = tensorize(X_train)
        X_test = tensorize(X_test)
        X_dev = tensorize(X_dev)

        for x in X_train: clean_tensor(x)
        for x in X_test: clean_tensor(x)
        for x in X_dev: clean_tensor(x)

        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        y_dev = torch.tensor(y_dev, dtype=torch.float32)

        if y_train.dim() == 3:
            y_train = y_train.squeeze(dim=-1)
            y_test = y_test.squeeze(dim=-1)
            y_dev = y_dev.squeeze(dim=-1)

        return X_train, X_test, X_dev, y_train, y_test, y_dev

    def get_data(self, shuffle=True, iterable=True, split='train'):
        x = self.datas[split]['X']
        y = self.datas[split]['y']
        feature_indexes = [_ind for _ind in self.feature_indexes]

        if 0 not in feature_indexes:
            feature_indexes = [0] + feature_indexes

        x = [_x for i, _x in enumerate(x) if i in feature_indexes]

        if iterable:
            all_tensors = list(x) + [y]
            dataset = TensorDataset(*all_tensors)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
            return loader
        else:
            return x, y