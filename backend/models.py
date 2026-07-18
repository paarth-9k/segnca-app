"""
Two models trained on the same real MRI slices, same loss, same optimizer budget,
so the comparison is fair:

1. SegNCA  — faithful reproduction of the Med-NCA segmentation rule (Kalkhof et al.,
   MICCAI/IPMI 2023). The input image channel is re-injected every step (never
   evolves), everything else — hidden channels + a segmentation-logit channel —
   evolves under one shared, tiny, local update rule.
2. TinyUNet — a standard encoder-decoder CNN, the architecture the paper benchmarks
   against, sized down to roughly the same task but with the usual global receptive
   field UNet gets from downsampling (which SegNCA deliberately does not have).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegNCA(nn.Module):
    def __init__(self, channel_n=16, input_channels=1, hidden_size=128, fire_rate=0.5):
        super().__init__()
        self.channel_n = channel_n
        self.input_channels = input_channels
        self.fire_rate = fire_rate
        self.fc0 = nn.Linear(channel_n * 3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()  # start as a near-identity / do-nothing rule

    def perceive(self, x):
        # x: (B, H, W, C) -- sobel x/y, depthwise, same trick as growing-NCA
        def conv(x, kernel):
            w = torch.tensor(kernel, dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
            w = w.repeat(self.channel_n, 1, 1, 1)
            xc = x.permute(0, 3, 1, 2)
            return F.conv2d(xc, w, padding=1, groups=self.channel_n).permute(0, 2, 3, 1)

        dx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        dy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        return torch.cat([x, conv(x, dx), conv(x, dy)], dim=-1)

    def update_step(self, x, fire_rate):
        y = self.perceive(x)
        dx = self.fc1(F.relu(self.fc0(y)))
        stochastic = (torch.rand(x.shape[0], x.shape[1], x.shape[2], 1, device=x.device) > fire_rate).float()
        return x + dx * stochastic

    def forward(self, x, steps=64, fire_rate=None):
        """x: (B, H, W, input_channels) real image; state is padded with zeros
        for the remaining channels internally."""
        fire_rate = self.fire_rate if fire_rate is None else fire_rate
        B, H, W, _ = x.shape
        pad = torch.zeros(B, H, W, self.channel_n - self.input_channels, device=x.device)
        state = torch.cat([x, pad], dim=-1)
        for _ in range(steps):
            new_state = self.update_step(state, fire_rate)
            # the image itself never evolves -- only the channels beyond it do
            state = torch.cat([state[..., :self.input_channels], new_state[..., self.input_channels:]], dim=-1)
        return state[..., -1]  # last channel = segmentation logit


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    """Standard 3-level UNet, channel width chosen so param count is in the same
    conversation as SegNCA is NOT the point -- the point is comparing a global
    receptive-field architecture against a purely-local one at realistic sizes."""
    def __init__(self, base=16):
        super().__init__()
        self.enc1 = ConvBlock(1, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        # x: (B, 1, H, W)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1).squeeze(1)  # logit, (B, H, W)


def dice_bce_loss(logits, target, eps=1e-6, pos_weight=15.0):
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1, 2))
    dice = 1 - (2 * inter + eps) / (probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + eps)
    pw = torch.tensor(pos_weight, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
    return dice.mean() + bce


def dice_score(logits, target, eps=1e-6):
    probs = (torch.sigmoid(logits) > 0.5).float()
    inter = (probs * target).sum(dim=(1, 2))
    return ((2 * inter + eps) / (probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + eps)).mean().item()


def iou_score(logits, target, eps=1e-6):
    probs = (torch.sigmoid(logits) > 0.5).float()
    inter = (probs * target).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) - inter
    return ((inter + eps) / (union + eps)).mean().item()
