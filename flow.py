import torch
from torch import nn, optim
from torch.functional import F
import numpy as np

####
# From Karpathy's MADE implementation
####

class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)        
        self.register_buffer('mask', torch.ones(out_features, in_features))
        
    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))
        
    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)

class MADE(nn.Module):
    def __init__(self, nin, hidden_sizes,
                 nout, num_masks=1, natural_ordering=False):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """
        
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"
        
        # define a simple MLP neural net
        self.net = []
        hs = [nin] + hidden_sizes + [nout]
        for h0,h1 in zip(hs, hs[1:]):
            self.net.extend([
                    MaskedLinear(h0, h1),
                    nn.ReLU(),
                ])
        self.net.pop() # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)
        
        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0 # for cycling through num_masks orderings
        
        self.m = {}
        self.update_masks() # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.
        
    def update_masks(self):
        if self.m and self.num_masks == 1: return # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)
        
        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks
        
        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin)
        for l in range(L):
            self.m[l] = rng.randint(self.m[l-1].min(), self.nin-1, size=self.hidden_sizes[l])
        
        # construct the mask matrices
        masks = [self.m[l-1][:,None] <= self.m[l][None,:] for l in range(L)]
        masks.append(self.m[L-1][:,None] < self.m[-1][None,:])
        
        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]]*k, axis=1)
        
        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l,m in zip(layers, masks):
            l.set_mask(m)
    
    def forward(self, x):
        return self.net(x)
      
      
####
# End Karpathy's code
####

class MAF(nn.Module):
    """x0 only depends on x0, etc"""
    def __init__(self, features, context, hidden=100, nlayers=1):
        super(self.__class__, self).__init__()
        self._fmualpha = MADE(features+context, [hidden]*nlayers, 2*(features+context), natural_ordering=True)
        self.context_map = nn.Linear(context, context)
        self.context = context
        self.features = features

    def fmualpha(self, x):
        # Only return the data parts: (conditioned on whole context vector)
        out = self._fmualpha(x)
        mu = out[:, self.context:self.context+self.features]
        alpha = out[:, 2*self.context+self.features:]
        return mu, alpha

    def load_context(self, x, context):
        return torch.cat((self.context_map(context), x), dim=1)

    def invert(self, u, context):
        _x = self.load_context(u, context)
        mu, alpha = self.fmualpha(_x)
        x = u * torch.exp(alpha) + mu
        return x

    def forward(self, x, context):
        # Invert the flow
        _x = self.load_context(x, context)
        mu, alpha = self.fmualpha(_x)
        u = (x - mu) * torch.exp(-alpha)
        log_det = - torch.sum(alpha, dim=1)
        return u, log_det


class Perm(nn.Module):
    def __init__(self, nvars, perm=None):
        super(self.__class__, self).__init__()
        # If perm is none, chose some random permutation that gets fixed at initialization
        if perm is None:
            perm = torch.randperm(nvars)
        self.perm = perm
        self.reverse_perm = torch.argsort(perm)

    def forward(self, x, context):
        idx = self.perm.to(x.device)
        return x[:, idx], 0

    def invert(self, x, context):
        rev_idx = self.reverse_perm.to(x.device)
        return x[:, rev_idx]
      
class Flow(nn.Module):
    def __init__(self, *layers):
        super(self.__class__, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, context):
        log_det = None
        for layer in self.layers:
            x, _log_det = layer(x, context)
            log_det = (log_det if log_det is not None else 0) + _log_det

        # Same ordering as input:
        for layer in self.layers[::-1]:
            if 'Perm' not in str(layer):
                continue
            x = x[:, layer.reverse_perm]
            
        return x, log_det

    def invert(self, u, context):
        for layer in self.layers:
            if 'Perm' not in str(layer):
                continue
            u = u[:, layer.perm]
            
            
        for layer in self.layers[::-1]:
            u = layer.invert(u, context)
        
        return u
