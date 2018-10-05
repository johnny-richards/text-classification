import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from global_attention import GlobalAttention

class RNNClassification(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, bidirectional, num_classes, p, pad_idx):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.hidden_factor = num_layers * (2 if bidirectional else 1)
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=p, bidirectional=bidirectional)
        self.linear = nn.Linear(self.hidden_factor * self.hidden_size, num_classes)
    def forward(self, inputs, length):
        batch_size, seq_len = inputs.size() 

        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        inputs = inputs[sorted_idx]

        inputs_emb = self.embedding(inputs)
        packed_input = rnn_utils.pack_padded_sequence(inputs_emb, sorted_lengths.data.tolist(), batch_first=True)
        _, hidden = self.rnn(packed_input)
        
        _, reverse_idx = torch.sort(sorted_idx)
        hidden = hidden[:, reverse_idx, :]

        hidden = hidden.transpose(0, 1).contiguous()
        hidden = hidden.view(batch_size, self.hidden_factor * self.hidden_size)

        output = self.linear(hidden)
        output = F.log_softmax(output)
        return output

class RNNClassificationAttention(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, bidirectional, num_classes, p, pad_idx):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.pad_idx = pad_idx
        self.hidden_factor = num_layers * (2 if bidirectional else 1)
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=p, bidirectional=bidirectional)
        self.linear = nn.Linear(self.hidden_factor * self.hidden_size, num_classes)

        # attention
        self.global_attn = GlobalAttention(self.hidden_factor * hidden_size, hidden_size * (2 if bidirectional else 1))

    def forward(self, inputs, length):
        batch_size, seq_len = inputs.size() 

        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        attn_mask = inputs == self.pad_idx
        inputs = inputs[sorted_idx]

        inputs_emb = self.embedding(inputs)
        packed_input = rnn_utils.pack_padded_sequence(inputs_emb, sorted_lengths.data.tolist(), batch_first=True)
        context, hidden = self.rnn(packed_input)

        padded_context = rnn_utils.pad_packed_sequence(context, batch_first=True)[0]
        padded_context = padded_context.contiguous()
        _, reverse_idx = torch.sort(sorted_idx)
        padded_context = padded_context[reverse_idx]
        hidden = hidden[:, reverse_idx, :]

        pb, ps, _ = padded_context.size()
        attn_mask = attn_mask[:, :ps]

        hidden = hidden.transpose(0, 1).contiguous()
        hidden = hidden.view(batch_size, self.hidden_factor * self.hidden_size)
        # attention
        hidden = self.global_attn(hidden, padded_context, attn_mask)

        output = self.linear(hidden)
        output = F.log_softmax(output)
        return output
