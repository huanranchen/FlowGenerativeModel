import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from optimizer import default_optimizer, default_lr_scheduler
from model import Glow


class FlowSolver():
    def __init__(self, student: Glow,
                 optimizer: torch.optim.Optimizer or None = None,
                 scheduler=None,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 eval_loader=None):
        self.student = student
        self.optimizer = optimizer if optimizer is not None else default_optimizer(self.student)
        self.scheduler = scheduler if scheduler is not None else default_lr_scheduler(self.optimizer)
        self.device = device

        # initialization
        self.init()

    def init(self):
        # change device
        self.student.to(self.device)

    def train(self,
              loader: DataLoader,
              total_epoch=2000,
              ):
        '''

        :param total_epoch:
        :param step_each_epoch: this 2 parameters is just a convention, for when output loss and acc, etc.
        :param fp16:
        :param generating_data_configuration:
        :return:
        '''
        self.student.train().to(self.device)
        for epoch in range(1, total_epoch + 1):
            train_loss = 0
            pbar = tqdm(loader)
            for step, (x, y) in enumerate(pbar, 1):
                x, y = x.to(self.device), y.to(self.device)
                loss = -self.student.log_likelihood(*self.student(x.view(x.shape[0], -1))).mean()
                train_loss += loss.item()
                self.optimizer.zero_grad()

                loss.backward()
                nn.utils.clip_grad_value_(self.student.parameters(), 0.1)
                self.optimizer.step()

                if step % 10 == 0:
                    pbar.set_postfix_str(f'loss={train_loss / step}')

            train_loss /= len(loader)

            print(f'epoch {epoch}, loss = {train_loss}')
            torch.save(self.student.state_dict(), 'student.pth')
