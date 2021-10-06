"""
Implements various flows.
Each flow is invertible so it can be forward()ed and backward()ed.
Notice that backward() is not backward as in backprop but simply inversion.
Each flow also outputs its log det J "regularization"

Reference:

Glow: Generative Flow with Invertible 1x1 Convolutions, Kingma and Dhariwal, Jul 2018
https://arxiv.org/abs/1807.03039

"Normalizing Flows for Probabilistic Modeling and Inference"
https://arxiv.org/abs/1912.02762
(review paper)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

#import sys
#sys.path.append('../')

#from nf import utils
#from nf.utils import torchutils

from nf.nets import MLP

class AffineConstantFlow(nn.Module):
    """
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """

    def __init__(self, dim,  scale=True, shift=True):
        super().__init__()

        self.s = (
            nn.Parameter(torch.randn(1, dim, requires_grad=True)) if scale else None
        )
        self.t = (
            nn.Parameter(torch.randn(1, dim, requires_grad=True)) if shift else None
        )

    def forward(self, x, context):

        s = self.s if self.s is not None else x.new_zeros(x.size())
        t = self.t if self.t is not None else x.new_zeros(x.size())
        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)

        return z, log_det

    def backward(self, z, context):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        t = self.t if self.t is not None else z.new_zeros(z.size())
        x = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=1) #do i not need a cuda here?!
        return x, log_det


class ActNorm(AffineConstantFlow):
    """
    Really an AffineConstantFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = False

    def forward(self, x, context):

        # first batch is used for init
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None  # for now
            self.s.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
            self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
            self.data_dep_init_done = True

        return super().forward(x, context)


class Invertible1x1Conv(nn.Module):
    """
    As introduced in Glow paper.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        Q = nn.init.orthogonal_(torch.randn(dim, dim))
        P, L, U = torch.lu_unpack(*Q.lu())
        self.P = P
        self.L = nn.Parameter(L)  # lower triangular portion
        self.S = nn.Parameter(U.diag())  # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(
            torch.triu(U, diagonal=1)
        )  # "crop out" diagonal, stored in S

    def _assemble_W(self):
        """ assemble W from its pieces (P, L, U, S) """
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim))
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, x, context):
        W = self._assemble_W()
        z = x @ W
        log_det = torch.sum(torch.log(torch.abs(self.S)))

        return z, log_det

    def backward(self, z, context):
        W = self._assemble_W()
        W_inv = torch.inverse(W)
        x = z @ W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.S)))
        
        return x, log_det


class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x, context):

        m, _ = x.shape
        log_det = torch.zeros(m)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x, context)
            log_det += ld

            zs.append(x)

            del ld

        return zs, log_det

    def backward(self, z, context):
        m, _ = z.shape
        log_det = torch.zeros(m)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z, context)
            log_det += ld
            xs.append(z)

            del ld

        return xs, log_det


class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows)

    def forward(self, x, context):

        zs, log_det = self.flow.forward(x, context=context)
        prior_logprob = self.prior.log_prob(zs[-1])
        prior_logprob = prior_logprob.view(x.size(0), -1).sum(1)

        return zs, prior_logprob, log_det

    def backward(self, z, context):

        xs, log_det = self.flow.backward(z, context=context)
        
        return xs, log_det

    def sample(self, num_samples, context):

        z = self.prior.sample((num_samples,)) 
        xs, _ = self.flow.backward(z, context=context)

        return xs