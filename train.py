import os
import json
import time
import torch
import numpy as np
import argparse
from collections import OrderedDict
from stsa import STSA
from model import RNNClassificationAttention
from torch.utils.data import DataLoader

def main(args):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
    splits = ['train', 'test']
    datasets = OrderedDict()
    for split in splits:
        datasets[split]=STSA(
            data_dir=os.path.join(args.data_dir, 'stsa.fine.{}.xls'.format(split)),
            split=split,
            min_occ=args.min_occ,
            max_seq_len=args.max_seq_len,
            vocab_dir=args.vocab_dir
        )
    model = RNNClassificationAttention(
        vocab_size = datasets['train'].vocab_size,
        embedding_size = args.embedding_size,
        hidden_size = args.hidden_size,
        num_layers = args.num_layers,
        bidirectional = args.bidirectional,
        num_classes = datasets['train'].num_cates,
        p = args.dropout_rate,
        pad_idx = datasets['train'].pad_idx
    )
    NLL = torch.nn.NLLLoss(size_average=False, ignore_index=datasets['train'].pad_idx)
    if torch.cuda.is_available():
        model.cuda()
    print(model)
    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        for split in splits:
            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle=split=='train',
                num_workers=4
            )
            if split=='train':
                model.train()
            else:
                model.eval()
            loss_tracker = []
            for iteration, batch in enumerate(data_loader):
                batch_size = batch['input'].size(0)
                for k, v in batch.items():
                    if torch.cuda.is_available():
                        v = v.cuda()
                output = model(batch['input'], batch['length'])
                label = batch['label']
                loss = NLL(output, label) / batch_size
                loss_tracker.append(loss.item())
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if iteration % args.print_every == 0 or (iteration + 1) == len(data_loader):
                    print('%s Batch %4d/%i Loss %9.4f' % (split.upper(), iteration, len(data_loader)-1, loss.item()))
            print('global loss {}'.format(np.mean(loss_tracker)))
            if split=='train':
                checkpoint_path = os.path.join(save_model_path, 'E%i.ckpt'%(epoch))
                torch.save(model.state_dict(), checkpoint_path)
                print('model saved at %s' % (checkpoint_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--vocab_dir', type=str, default='fine.vocab')
    parser.add_argument('--max_seq_len', type=int, default=60)
    parser.add_argument('--min_occ', type=int, default=1)

    parser.add_argument('-ep', '--epochs', type=int, default=70)
    parser.add_argument('-bs', '--batch_size', type=int, default=512)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-dp', '--dropout_rate', type=float, default=0.5)

    parser.add_argument('-eb', '--embedding_size', type=int, default=16)
    parser.add_argument('-hs', '--hidden_size', type=int, default=16)
    parser.add_argument('-nl', '--num_layers', type=int, default=2)
    parser.add_argument('-bi', '--bidirectional', action='store_true')

    parser.add_argument('-v','--print_every', type=int, default=1)
    parser.add_argument('-bin','--save_model_path', type=str, default='bin')
    args = parser.parse_args()

    main(args)
