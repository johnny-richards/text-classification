import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalAttention(nn.Module):
    def __init__(self, qsize, ksize):
        super().__init__()
        self.semtrans = nn.Linear(ksize, qsize)
        self.sembacks = nn.Linear(2 * qsize, qsize)
    def forward(self, query, context, mask):
# context = batch x seqk x ksize
# query = batch x qsize
        new_query = query.unsqueeze(1)
        affine_context = self.semtrans(context)
# affine_context = batch x seqk x qsize
        affine_context_t = affine_context.transpose(1,2).contiguous()
# affine_context_t = batch x qsize x seqk
# mask = batch x seqk
        mask = mask.unsqueeze(1)
# mask = batch x 1 x seqk
        weight = torch.bmm(new_query, affine_context_t)
# weight = batch x seq x seqk
        mask = mask.expand_as(weight)
# mask = batch x seq x seqk
        weight = weight.masked_fill_(mask, -1e18)
        weight = F.softmax(weight, dim=2)
# weight = batch x seq x seqk
        weighted_context = torch.bmm(weight, affine_context)
# weighted_context = batch x seq x qsize
        output = torch.cat([weighted_context, new_query], 2)
# output = batch x seq x (2 x qsize)
        output = self.sembacks(output)
# output = batch x seq x qsize
        output = F.tanh(output)
        output = output.squeeze()
        return output
