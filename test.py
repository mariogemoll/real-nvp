import torch
from models import RealNVP, RealNVPLoss

model = RealNVP(num_scales=2, in_channels=3, mid_channels=64, num_blocks=2)
loss_fn = RealNVPLoss()

x = torch.rand(4, 3, 32, 32)
z, sldj = model(x)
loss = loss_fn(z, sldj)
loss.backward()
reversed, _ = model(z, reverse=True)
