import torch
from torch.utils.data import DataLoader
import pytorch_fid
import os
from typing import Callable, Iterable
from torchvision import transforms
from torch import nn
from tqdm import tqdm


@torch.no_grad()
def test_fid(loader: DataLoader or Iterable, generator: Callable,
             path: str = './test_fid/',
             dataset_sample_images: int = 50000,
             total_generation_iter: int = 100):
    to_img = transforms.ToPILImage()
    if not os.path.exists(path):
        os.mkdir(path)
    dataset_path = os.path.join(path, 'dataset/')
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    generated_img_path = os.path.join(path, 'generated/')
    if not os.path.exists(generated_img_path):
        os.mkdir(generated_img_path)

    # extract images from loader
    imgs = []
    for x, _ in loader:
        imgs += list(torch.split(x, split_size_or_sections=1, dim=0))
    imgs = imgs[:dataset_sample_images]
    for i, img in enumerate(imgs):
        img = to_img(img.squeeze())
        img.save(os.path.join(dataset_path, f'{i}.png'))
    # extract images from generator
    imgs.clear()
    for _ in tqdm(range(total_generation_iter)):
        imgs += list(torch.split(generator(), split_size_or_sections=1, dim=0))
    for i, img in enumerate(imgs):
        img = to_img(img.squeeze())
        img.save(os.path.join(generated_img_path, f'{i}.png'))

    # test fid
    os.system(f'python -m pytorch_fid {dataset_path} {generated_img_path}')
