# MADE & MAF implementation in PyTorch

Simple normalizing flow with a conditional variable of any size.

Example:

Make the model, for 3 features (+1 conditional).
```python
m = Flow(
    *[[MAF(3, 1), Perm()][i%2] for i in range(5*2 + 1)]
).cuda()

```
This has a stack of 5 MADEs, permuting
the dependencies between each. The permutation is randomly
chosen at initialization.

Let's say our data is:

z ~ U(0, 5)

x[i] ~ N(z, 1) for i=1:3

Let's train it with a base distribution of a unit Gaussian.
```python

opt = optim.Adam(m.parameters(), 1e-3)

for i in trange(30000):
    conditional = torch.rand(32, 1, device='cuda') * 5
    x = torch.randn(32, 3, device='cuda') + conditional

    u, log_det = m(x, conditional)

    log_prob = -u.pow(2).sum(1)/2
    normalized_log_prob = log_prob + log_det

    loss = -normalized_log_prob.mean()
    loss.backward()
    nn.utils.clip_grad_norm_(m.parameters(), 1)
    opt.step()
    if i % (30000//100) == 0:
        print(loss.item())
```

Test it:
```python

conditional = torch.rand(32, 1, device='cuda') * 5

#Input noise:
u = torch.randn(32, 3, device='cuda')

(m.invert(u, conditional) - conditional).std()
```
^ About 1, which is what we want.
