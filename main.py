from data import get_mnist_train
from solver import FlowSolver
from model import Glow

model = Glow(32 * 32)
loader = get_mnist_train()
solver = FlowSolver(model)
solver.train(loader)
