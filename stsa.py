import torch
import json
import numpy as np
from torch.utils.data import Dataset

class STSA(Dataset):
    def __init__(self, data_dir, split, min_occ, max_seq_len, vocab_dir):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.data_dir = data_dir
        self.vocab_dir = vocab_dir
        self.vocab = {'pad': 0, 'unk': 1}
        self.data = []
        self.min_occ = min_occ
        self.split = split
        if split=='train':
            self._create_vocab()
        else:
            self._load_vocab()
        self._create_data()
    @property
    def pad_idx(self):
        return self.vocab.get('pad', 0)
    @property
    def vocab_size(self):
        return len(self.vocab)
    @property
    def num_cates(self):
        return self.num_categories
    def _create_vocab(self):
        vcount = {}
        with open(self.data_dir, 'r') as f:
            for lines in f:
                line = lines.strip().split(' ')
                texts = line[1:]
                for text in texts:
                    if not text in vcount:
                        vcount[text] = 1
                    else:
                        vcount[text] += 1
        with open(self.data_dir, 'r') as f:
            for lines in f:
                line = lines.strip().split(' ')
                texts = line[1:]
                for text in texts:
                    if vcount[text] < self.min_occ:
                        continue
                    if not text in self.vocab:
                        self.vocab[text] = len(self.vocab)
        print('vocabulary created size is: {}'.format(len(self.vocab)))
        with open(self.vocab_dir, 'w') as f:
            f.write(json.dumps(self.vocab))
    def _load_vocab(self):
        with open(self.vocab_dir, 'r') as f:
            self.vocab = json.loads(f.readline())
        print('vocabulary loaded size is: {}'.format(len(self.vocab)))
    def _create_data(self):
        cate_dict = {}
        max_label = -1
        with open(self.data_dir, 'r') as f:
            for lines in f:
                line = lines.strip().split(' ')
                texts = line[1:]
                cur_batch = {'length': len(texts), 'label': int(line[0])}
                max_label = max(max_label, int(line[0]))
                if line[0] not in cate_dict:
                    cate_dict[line[0]] = int(line[0])
                cur_input = []
                for text in texts:
                    cur_id = self.vocab.get(text, self.vocab['unk'])
                    cur_input.append(cur_id)
                cur_input.extend((self.max_seq_len - len(cur_input)) * [self.vocab['pad']])
                cur_batch['input']=cur_input
                self.data.append(cur_batch)
        self.num_categories = len(cate_dict)
        assert (max_label + 1) == len(cate_dict)
        print('{} data created, total number is {}'.format(self.split, len(self.data)))
    def __getitem__(self, index):
        return {
                'input': np.asarray(self.data[index]['input']),
                'label': np.asarray(self.data[index]['label']),
                'length': np.asarray(self.data[index]['length'])
        }
    def __len__(self):
        return len(self.data)
