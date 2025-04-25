import torch
from models import RealNVP, RealNVPLoss

model = RealNVP(num_scales=2, in_channels=3, mid_channels=64, num_blocks=2)
loss_fn = RealNVPLoss()

x = torch.rand(4, 3, 32, 32)
z, sldj = model(x)
loss = loss_fn(z, sldj)
loss.backward()

model.eval()

with torch.no_grad():
    samples, _ = model(torch.randn((4, 3, 32, 32)), reverse=True)
    samples = torch.sigmoid(samples)
    samples_z, sldj = model(samples)

