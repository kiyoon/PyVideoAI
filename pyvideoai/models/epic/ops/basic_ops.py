# Kiyoon modifications:
# The autograd Function was using deprecated PyTorch implementation, so modified to work with newer PyTorch.
# Also, AMP (automatic mixed precision) is disabled for this autograd function and will use 32-bit.

import torch
from torch.cuda.amp import custom_fwd, custom_bwd       # disable AMP and force to 32-bit


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


#class SegmentConsensus(torch.autograd.Function):
#    def __init__(self, consensus_type, dim=1):
#        self.consensus_type = consensus_type.lower()
#        self.dim = dim
#        self.shape = None
#
#    @staticmethod
#    def forward(self, input_tensor):
#        self.shape = input_tensor.size()
#        if self.consensus_type == "avg":
#            output = input_tensor.mean(dim=self.dim, keepdim=True)
#        elif self.consensus_type == "max":
#            output, _ = input_tensor.max(dim=self.dim, keepdim=True)
#        elif self.consensus_type == "identity":
#            output = input_tensor
#        else:
#            output = None
#
#        return output
#
#    @staticmethod
#    def backward(self, grad_output):
#        if self.consensus_type == "avg":
#            grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
#        elif self.consensus_type == "identity":
#            grad_in = grad_output
#        else:
#            grad_in = None
#
#        return grad_in


class SegmentConsensusAvg(torch.autograd.Function):
    dim=1

    @staticmethod
    #@custom_fwd(cast_inputs=torch.float32)
    @custom_fwd
    def forward(ctx, input_tensor):
        ctx.shape = input_tensor.size()
        output = input_tensor.mean(dim=SegmentConsensusAvg.dim, keepdim=True)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grad_in = grad_output.expand(ctx.shape) / float(ctx.shape[SegmentConsensusAvg.dim])

        return grad_in


class SegmentConsensusMax(torch.autograd.Function):
    dim=1

    @staticmethod
    #@custom_fwd(cast_inputs=torch.float32)
    @custom_fwd
    def forward(ctx, input_tensor):
        output = input_tensor.max(dim=SegmentConsensusMax.dim, keepdim=True)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return None


class SegmentConsensusIdentity(torch.autograd.Function):

    @staticmethod
    #@custom_fwd(cast_inputs=torch.float32)
    @custom_fwd
    def forward(ctx, input_tensor):
        return input_tensor 

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output


class ConsensusModule(torch.nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != "rnn" else "identity"
        self.dim = dim

        assert dim == 1, "only support dim=1"

        if self.consensus_type == "avg":
            self.segment_consensus = SegmentConsensusAvg
        elif self.consensus_type == "max":
            self.segment_consensus = SegmentConsensusMax
        elif self.consensus_type == "identity":
            self.segment_consensus = SegmentConsensusIdentity
        else:
            raise ValueError("Undefined segment consensus type: {}".format(consensus_type))
        #self.segment_consensus = SegmentConsensus(self.consensus_type, self.dim)

    def forward(self, input):
        return self.segment_consensus.apply(input)
