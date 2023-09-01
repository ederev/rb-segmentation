import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import filterfalse as ifilterfalse


def hard_sigmoid(input):
    return F.relu6(input + 3.0) / 6.0


class HSigmoid(nn.Module):
    def forward(self, input):
        return hard_sigmoid(input)


class HSwish(nn.Module):
    def forward(self, input):
        return input * hard_sigmoid(input)


class SELayer(nn.Module):
    def __init__(self, c_in, reduction=4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            HSigmoid(),
        )

    def forward(self, input):
        output = input.mean((2, 3))
        output = self.layers(output)[:, :, None, None]
        return input * output


class Residual(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, input):
        return input + super().forward(input)


def residual_block(c_in, conv, norm, act):
    return Residual(
        conv(c_in, 2 * c_in, 1, stride=1),
        norm(2 * c_in),
        act,
        conv(c_in * 2, c_in * 2, 3, stride=1, padding=1, groups=c_in * 2),
        norm(2 * c_in),
        act,
        SELayer(c_in * 2),
        conv(c_in * 2, c_in, 1, stride=1),
        norm(c_in),
    )


def down_block(conv=nn.Conv2d, norm=nn.BatchNorm2d, act=HSwish()):
    def make_block(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False),
            norm(c_out),
            act,
            residual_block(c_out, conv, norm, act),
        )

    return make_block


def up_block(conv=nn.Conv2d, norm=nn.BatchNorm2d, act=HSwish()):
    def make_block(c_in, c_out):
        return nn.Sequential(
            residual_block(c_in, conv, norm, act),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            norm(c_out),
            act,
        )

    return make_block


class DownBranch(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, input):
        out = [input]
        for module in self._modules.values():
            out.append(module(out[-1]))
        return out[1:]


class UpBranch(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, input):
        y = torch.empty(0, dtype=input[0].dtype, device=input[0].device)
        for module in self._modules.values():
            y = module(torch.cat((input.pop(), y), 1))
        return y


class UNetCustom(nn.Module):
    def __init__(
        self, down_block, up_block, min_channels, max_channels, depth, n_classes
    ):
        super().__init__()
        ch_down, ch_up = self.compute_channels(min_channels, max_channels, depth)
        self.pre = nn.Conv2d(3, min_channels, 1)
        self.down = DownBranch(*map(down_block, *ch_down))
        self.up = UpBranch(*map(up_block, *ch_up))
        self.post = nn.Conv2d(min_channels, n_classes, 1)

    @staticmethod
    def compute_channels(min_channels, max_channels, depth):
        channels = [
            min(max_channels, min_channels * (2**d)) for d in range(depth + 1)
        ]
        down = channels[:-1], channels[1:]
        c_in = [2 * c for c in channels[1:]]
        c_in[-1] = channels[-1]  # no skip connection for first up_block
        up = reversed(c_in), reversed(channels[:-1])
        return down, up

    def forward(self, input):
        output = self.pre(input)
        output = self.down(output)
        output = self.up(output)
        output = self.post(output)
        return output


def focal_loss(yhat, y, gamma=2):
    logpt = -F.cross_entropy(yhat.flatten(2), y.flatten(1))
    loss = -((1 - logpt.exp()) ** gamma) * logpt
    return loss


def lovasz_grad(gt_sorted):
    # Computes gradient of the Lovasz extension w.r.t sorted errors
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes="present", per_image=False, ignore=None):
    if per_image:
        loss = mean(
            lovasz_softmax_flat(
                *flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore),
                classes=classes
            )
            for prob, lab in zip(probas, labels)
        )
    else:
        loss = lovasz_softmax_flat(
            *flatten_probas(probas, labels, ignore), classes=classes
        )
    return loss


def lovasz_softmax_flat(probas, labels, classes="present"):
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if classes is "present" and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError("Sigmoid output possible only with 1 class")
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    # flattens predictions in the batch
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    # nanmean compatible with generators
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def dice_loss(preds, target):
    prob = preds.softmax(1)
    target = (
        F.one_hot(target, num_classes=prob.size(1)).permute(0, 3, 1, 2).contiguous()
    )
    prob = prob[:, 1:, ...]
    target = target[:, 1:, ...]
    inter = 2 * torch.sum(prob * target)
    union = torch.sum(prob + target)
    return 1 - (inter + 1) / (union + 1)
