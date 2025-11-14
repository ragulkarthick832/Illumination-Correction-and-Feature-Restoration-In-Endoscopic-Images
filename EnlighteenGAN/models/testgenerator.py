import torch
from generator import GeneratorUNet

G = GeneratorUNet().cuda()
x = torch.randn(1,9,256,256).cuda()
y = G(x)
print(y.shape)
