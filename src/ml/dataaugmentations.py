import os
import zarr
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
torch.manual_seed(0)
import torchvision.transforms.functional as tf
import random

class PermeabilityTransform:
    def __init__(self, enable_shift=True):
        self.enable_shift = enable_shift
        self.permutations = [
            ('I', lambda img: img, lambda K: K),
            ('R', lambda img: tf.rotate(img, 90),
                  lambda K: torch.tensor([K[3], -K[2], -K[1], K[0]])),  # 90°
            ('RR', lambda img: tf.rotate(img, 180),
                   lambda K: torch.tensor([K[0], K[1], K[2], K[3]])),  # 180°
            ('RRR', lambda img: tf.rotate(img, 270),
                    lambda K: torch.tensor([K[3], -K[2], -K[1], K[0]])),  # 270°
            ('H', lambda img: tf.hflip(img),
                  lambda K: torch.tensor([K[0], -K[1], -K[2], K[3]])),  # Horizontal flip
            ('HR', lambda img: tf.rotate(tf.hflip(img), 90),
                   lambda K: torch.tensor([K[3], K[2], K[1], K[0]])),  # H + 90°
            ('HRR', lambda img: tf.rotate(tf.hflip(img), 180),
                    lambda K: torch.tensor([K[0], -K[1], -K[2], K[3]])),  # H + 180°
            ('HRRR', lambda img: tf.rotate(tf.hflip(img), 270),
                     lambda K: torch.tensor([K[3], K[2], K[1], K[0]])),  # H + 270°
        ]
    def periodic_shift(self, img):
        sx = random.randint(0, img.shape[-2] - 1)
        sy = random.randint(0, img.shape[-1] - 1)
        return torch.roll(img, shifts=(sx, sy), dims=(-2, -1))

    def __call__(self, image, K):
        # Randomly choose one of the 8 permutations
        _, img_transform, k_transform = random.choice(self.permutations)
        if self.enable_shift:
            image = self.periodic_shift(image)
        image = img_transform(image)
        K = k_transform(K)
        return image, K


class DispersionTransform:
    def __init__(self, enable_shift=True):
        self.enable_shift = enable_shift
        self.permutations = [
            lambda img, Dx, Dy: (img, torch.tensor([Dx[0],Dx[3]])),
            lambda img, Dx, Dy: (tf.rotate(img, 90), torch.tensor([Dy[3],Dy[0]])),
            lambda img, Dx, Dy: (tf.rotate(tf.hflip(img), 180), torch.tensor([Dx[0],Dx[3]])),
            lambda img, Dx, Dy: (tf.rotate(tf.hflip(img), 270), torch.tensor([Dy[3],Dy[0]])),
        ]
    def periodic_shift(self, img):
        sx = random.randint(0, img.shape[-2] - 1)
        sy = random.randint(0, img.shape[-1] - 1)
        return torch.roll(img, shifts=(sx, sy), dims=(-2, -1))

    def __call__(self, image, Dx, Dy):
        if self.enable_shift:
            image = self.periodic_shift(image)
        t = random.choice(self.permutations)
        return t(image, Dx, Dy)