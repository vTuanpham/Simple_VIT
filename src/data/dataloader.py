import os
import sys
import random
import warnings
from typing import Union, List
sys.path.insert(0, r'./')

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, random_split
import torchvision.transforms as transforms

from src.utils.util_funcs import timeit, plot_image, TwoWayDict


CLASSES = TwoWayDict({"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4,
                      "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9}).dict


class CIFAR(Dataset):
    def __init__(self, transform, datapath: str, labelpath: str):
        self.transform = transform
        self.labels = pd.read_csv(labelpath, delimiter=',')
        self.data = sorted(os.listdir(datapath), key=len)
        self.root_dataset = [os.path.join(datapath, example_path) for example_path in self.data]

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def label_map(keys) -> Union[List[int], List[str]]:
        return [CLASSES[int(key)] for key in keys] if isinstance(keys, torch.Tensor) else CLASSES[keys]

    def __getitem__(self, idx):
        img_url = self.root_dataset[idx]
        label = self.labels['label'][idx]

        try:
            img = Image.open(img_url).convert('RGB')
        except IOError:
            warnings.warn("IO Error, watch out !")
            raise "IO Error!"

        if self.transform:
            img = self.transform(img)

        return img, self.label_map(label)


class CIFARDataloader:
    def __init__(self, transform,
                 datapath: str,
                 labelpath: str,
                 train_size: float,
                 seed: int,
                 batch_size: int,
                 num_worker: int=2):
        self.transform = transform
        self.datapath = datapath
        self.labelpath = labelpath

        self.seed = seed
        self.batch_size = batch_size
        self.train_size = train_size

        self.num_worker = num_worker
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

    @timeit
    def __call__(self) -> dict:
        train_eval_dataset = CIFAR(self.transform, self.datapath, self.labelpath)
        train_dataset, eval_dataset = random_split(train_eval_dataset, lengths=[self.train_size, 1-self.train_size])

        train_dataloader = self.get_dataloader(train_dataset, shuffle_flag=True,
                                               batch_size=self.batch_size)
        eval_dataloader = self.get_dataloader(eval_dataset, shuffle_flag=False,
                                               batch_size=self.batch_size)

        print(f"\n Data init log:"
              f"\n Batch size: {self.batch_size}"
              f"\n Num train batches: {len(train_dataloader)}"
              f"\n Num eval batches: {len(eval_dataloader)}"
              f"\n Num train examples: {len(train_dataloader.dataset)}"
              f"\n Eval examples: {len(eval_dataloader.dataset)}"
              f"\n Seed: {self.seed}"
              f"\n Num worker: {self.num_worker}")

        return {"train": train_dataloader, "eval": eval_dataloader}

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_dataloader(self, dataset, shuffle_flag: bool = False, batch_size: int = 1) -> DataLoader:
        sampler = RandomSampler(data_source=dataset, generator=self.generator) if shuffle_flag else \
            SequentialSampler(dataset)
        return DataLoader(dataset,
                          sampler=sampler,
                          batch_size=batch_size,
                          drop_last=True,
                          num_workers=self.num_worker,
                          worker_init_fn=self.seed_worker,
                          pin_memory=torch.cuda.is_available())


if __name__ == "__main__":
    dataloader_args = {
        "datapath": r'./src/data/train',
        "labelpath": r'./src/data/trainLabels.csv',
        "batch_size": 5,
        "transform": transforms.Compose([
            transforms.RandomCrop(32, padding=1, pad_if_needed=True),
            transforms.Resize(224),
            transforms.ToTensor()
        ]),
        "train_size": 0.8,
        "seed": 42
    }
    dataloaders = CIFARDataloader(**dataloader_args)
    dataloaders = dataloaders.__call__()

    for image, label in iter(dataloaders['train']):
        print(label)
        plot_image(image, CIFAR.label_map(label))
        break

