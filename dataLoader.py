# Part of this file is derived from
# https://github.com/albertyang33/FOREST

import numpy as np
from torch.utils.data import Dataset
import pickle
import os

class DataOption:
    def __init__(self, data_name='christianity'):
        self.net_data = 'data/' + data_name + '/edges.txt'
        self.idx2u_dict = 'data/' + data_name + '/idx2u.pickle'
        self.u2idx_dict = 'data/' + data_name + '/u2idx.pickle'
        self.cascades_data = 'data/' + data_name + '/cascades.txt'
        self.save_path = ''
        self.embed_dim = 64

class Cascades(Dataset):
    def __init__(self, dataset, max_seq_length, mode="train"):
        train, valid, test = Split_data(data_name=dataset, train_rate=0.8, valid_rate=0.1, load_dict=True)
        if mode == 'train':
            cascades, times = train
        elif mode == 'valid':
            cascades, times = valid
        else:
            cascades, times = test
        examples, examples_times = get_data_set(cascades, times, max_len=max_seq_length, mode=mode)
        self.examples, self.lengths, self.targets, self.masks, self.examples_times, self.targets_times = \
            prepare_sequences(examples, examples_times, max_len=max_seq_length, mode=mode)

    def __getitem__(self, index):
        examples = self.examples[index]
        targets = self.targets[index]
        masks = self.masks[index]
        return examples, targets, masks

    def __len__(self):
        return len(self.examples)


def get_cascades(data_name, load_dict):
    options = DataOption(data_name)
    with open(options.u2idx_dict, 'rb') as f:
        u2idx_dict = pickle.load(f)
    with open(options.idx2u_dict, 'rb') as f:
        idx2u_dict = pickle.load(f)
    user_size = len(u2idx_dict)

    '''读取级联数据集'''
    cascades = []
    timestamps = []
    if os.path.exists(options.cascades_data):
        with open(options.cascades_data) as f:
            for line in f:
                if (len(line.strip()) == 0):
                    continue

                timestamp_list = []
                user_list = []
                chunks = line.strip().split(',')
                for chunk in chunks:
                    if chunk == '':
                        break
                    # Twitter, Douban
                    if (len(chunk.split()) == 2):
                        user, timestamp = chunk.split()
                    # Android, Christianity
                    if (len(chunk.split()) == 3):
                        rootuser, user, timestamp = chunk.split()
                        if rootuser in u2idx_dict:
                            user_list.append(u2idx_dict[rootuser])
                            timestamp_list.append(float(timestamp))

                    if user in u2idx_dict:
                        user_list.append(u2idx_dict[user])
                        timestamp_list.append(float(timestamp))
                if len(user_list) > 1 and len(user_list) <= 500:
                    cascades.append(user_list)
                    timestamps.append(timestamp_list)

    return cascades, timestamps, user_size

def get_data_set(cascades, timestamps, max_len=None, test_min_percent=0.1, test_max_percent=0.5, mode='test'):
    """ Create train/val/test examples from input cascade sequences. Cascade sequences are truncated based on max_len.
    Test examples are sampled with seed set percentage between 10% and 50%. Train/val sets include examples of all
    possible seed sizes. """
    dataset, dataset_times = [], []
    eval_set, eval_set_times = [], []
    for cascade in cascades:
        if max_len is None or len(cascade) < max_len:
            dataset.append(cascade)
        else:
            dataset.append(cascade[0:max_len])  # truncate

    for ts_list in timestamps:
        if max_len is None or len(ts_list) < max_len:
            dataset_times.append(ts_list)
        else:
            dataset_times.append(ts_list[0:max_len])  # truncate

    for cascade, ts_list in zip(dataset, dataset_times):
        assert len(cascade) == len(ts_list)

        for j in range(2, len(cascade)):
        # for j in range(1, len(cascade)):
            seed_set = cascade[0:j]
            seed_set_times = ts_list[0:j]
            remain = cascade[j:]
            remain_times = ts_list[j:]
            seed_set_percent = len(seed_set) / (len(seed_set) + len(remain))
            if mode == 'train' or mode == 'valid':
                eval_set.append((seed_set, remain))
                eval_set_times.append((seed_set_times, remain_times))
            if mode == 'test' and (test_min_percent < seed_set_percent < test_max_percent):
                eval_set.append((seed_set, remain))
                eval_set_times.append((seed_set_times, remain_times))
    print("# {} examples {}".format(mode, len(eval_set)))
    return eval_set, eval_set_times

def prepare_sequences(examples, examples_times, max_len=None, cascade_batch_size=1, mode='train'):
    """ Prepare sequences by padding and adding dummy evaluation sequences. """
    seqs_x = list(map(lambda seq_t: (seq_t[0][(-1) * max_len:], seq_t[1]), examples))
    times_x = list(map(lambda seq_t: (seq_t[0][(-1) * max_len:], seq_t[1]), examples_times))
    # add padding.
    lengths_x = [len(s[0]) for s in seqs_x]
    lengths_y = [len(s[1]) for s in seqs_x]

    if len(seqs_x) % cascade_batch_size != 0 and (mode == 'test' or mode == 'val'):
        # Dummy sequences for evaluation: this is required to ensure that each batch is full-sized -- else the
        # data may not be split perfectly while evaluation.
        x_batch_size = (1 + len(seqs_x) // cascade_batch_size) * cascade_batch_size
        lengths_x.extend([1] * (x_batch_size - len(seqs_x)))
        lengths_y.extend([1] * (x_batch_size - len(seqs_x)))

    x_lengths = np.array(lengths_x).astype('int32')
    max_len_x = max_len
    # mask input with start token (n_nodes + 1) to work with embedding_lookup
    start_token = 0
    x = np.ones((len(lengths_x), max_len_x)).astype('int32') * start_token
    # mask target with -1 so that tf.one_hot will return a zero vector for padded nodes
    y = np.ones((len(lengths_y), max_len_x)).astype('int32') * -1
    # activation times are set to vector of ones.
    x_times = np.ones((len(lengths_x), max_len_x)).astype('int32') * -1
    y_times = np.ones((len(lengths_y), max_len_x)).astype('int32') * -1
    mask = np.ones_like(x)

    # Assign final set of sequences.
    for idx, (s_x, t) in enumerate(seqs_x):
        end_x = lengths_x[idx]
        end_y = lengths_y[idx]
        x[idx, :end_x] = s_x
        y[idx, :end_y] = t
        mask[idx, end_x:] = 0

    for idx, (s_x, t) in enumerate(times_x):
        end_x = lengths_x[idx]
        end_y = lengths_y[idx]
        x_times[idx, :end_x] = s_x
        y_times[idx, :end_y] = t

    return x, x_lengths, y, mask, x_times, y_times


def Split_data(data_name, train_rate=0.8, valid_rate=0.1, load_dict=True):
    cascades, timestamps, user_size = get_cascades(data_name, load_dict)

    '''data split'''
    # train
    train_idx_ = int(train_rate*len(cascades))
    train = cascades[0:train_idx_]
    train_t = timestamps[0:train_idx_]
    train = [train, train_t]

    # valid
    valid_idx_ = int((train_rate+valid_rate)*len(cascades))
    valid = cascades[train_idx_:valid_idx_]
    valid_t = timestamps[train_idx_:valid_idx_]
    valid = [valid, valid_t]

    # test
    test = cascades[valid_idx_:]
    test_t = timestamps[valid_idx_:]
    test = [test, test_t]

    # print info
    total_len = sum([len(i) for i in cascades])
    train_size = len(train_t)
    valid_size = len(valid_t)
    test_size = len(test_t)
    print(f"train size: {train_size}\n valid size: {valid_size}\n test size: {test_size}\n")
    print(f'number of cascades: {len(cascades)}')
    print(f'number of user: {user_size-2}')
    print(f'average length of cascades: {total_len/len(cascades)}')
    print(f'max length of cascades: {max(len(i) for i in cascades)}')
    print(f'min length of cascades: {min(len(i) for i in cascades)}')

    return train, valid, test

def buildIndex(cascades_data):
    user_set = set()
    u2idx_dict = {}
    idx2u_dict = []

    for line in open(cascades_data):
        if len(line.strip()) == 0:
            continue
        chunks = line.strip().split(',')
        for chunk in chunks:
            if len(chunk.split()) == 2:
                user, timestamp = chunk.split()
            elif len(chunk.split()) == 3:
                root_user, user, timestamp = chunk.split()
                user_set.add(root_user)
            user_set.add(user)

    pos = 0
    u2idx_dict['<blank>']= pos
    idx2u_dict.append('<blank>')
    pos += 1
    u2idx_dict['</s>'] = pos
    idx2u_dict.append('</s>')
    pos += 1

    for user in user_set:
        u2idx_dict[user] = pos
        idx2u_dict.append(user)
        pos += 1
    user_size = len(user_set)
    print(f'user size: {user_size}')

    return user_size, u2idx_dict, idx2u_dict