import torch
from data import get_mnist_train
from solver import FlowSolver
from model import Glow
from utils.ImageHandling import save_image

model = Glow(32 * 32)
model.load_state_dict(torch.load(('./student.pth')))

loader = get_mnist_train()
# solver = FlowSolver(model)
# solver.train(loader)

# from model.glow.glow import Block
# #
# model = Block(32 * 32)

model.eval().cuda()
x = torch.randn((1, 32 * 32)).cuda()
ori_x = x.clone()
x = model.inverse(x)
print(torch.sum((model(x)[0] - ori_x) ** 2))
x = x.view(1, 1, 32, 32)

x = x / 2 + 0.5
x = torch.clamp(x, min=0, max=1)
print(x)
save_image(x)
