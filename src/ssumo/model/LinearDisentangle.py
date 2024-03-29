from torch.autograd import Function
import torch.nn as nn
import torch

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -alpha * grad_output
        return grad_input, None

revgrad = GradientReversal.apply

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha):
        super(GradientReversalLayer, self).__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)


class ReversalEnsemble(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ReversalEnsemble, self).__init__()

        self.lin = nn.Linear(in_dim, out_dim)

        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim)
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, out_dim),
        )

    def forward(self, z):
        a = self.lin(z)
        b = self.mlp1(z)
        c = self.mlp2(z)
        d = self.mlp3(z)
        return a, b, c, d

class LinearDisentangle(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False, reversal="linear", alpha=1.0, do_detach=True):
        super(LinearDisentangle, self).__init__()
        self.do_detach = do_detach

        self.decoder = nn.Linear(in_dim, out_dim, bias=bias)
        if reversal == "mlp":
            self.reversal = nn.Sequential(
                GradientReversalLayer(alpha),
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, out_dim),
            )
        elif reversal == "linear":
            self.reversal = nn.Sequential(
                GradientReversalLayer(alpha), nn.Linear(in_dim, out_dim, bias=True)
            )
        elif reversal == "ensemble":
            self.reversal = nn.Sequential(
                GradientReversalLayer(alpha), ReversalEnsemble(in_dim, out_dim)
            )
        else:
            self.reversal = None

    def forward(self, z):
        x = self.decoder(z)

        if self.reversal is not None:
            if self.do_detach:
                w = self.decoder.weight.detach()
            else:
                w = self.decoder.weight

            nrm = w @ w.T
            if self.do_detach:
                z_sub = z - torch.linalg.solve(nrm, x.detach().T).T @ w
            else:
                z_sub = z - torch.linalg.solve(nrm, x.T).T @ w

            return x, self.reversal(z_sub)
        
        return x, None
    
    # def forward(self, z):
    #     x = self.decoder(z)

    #     if self.do_detach:
    #         w = self.decoder.weight.detach()
    #     else:
    #         w = self.decoder.weight

    #     nrm = (w @ w.T).ravel()
1
    #     if self.do_detach:
    #         z_sub = z - (x.detach() @ w) / nrm
    #     else:
    #         z_sub = z - (x @ w) / nrm
    #     return x, self.reversal(z_sub)
