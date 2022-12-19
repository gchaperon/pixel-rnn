import os
import torch
import torchvision
import torchvision.transforms as transforms
import typing as tp
import pytorch_lightning as pl


class LitMNIST(pl.LightningDataModule):
    datadir: str
    batch_size: int

    def __init__(self, datadir: str, batch_size: int) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("datadir",))

        self.datadir = datadir
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        torchvision.datasets.MNIST(root=self.datadir, train=True, download=True)
        torchvision.datasets.MNIST(root=self.datadir, train=False, download=True)

    def setup(self, stage: tp.Literal["train", "test"] | None = None) -> None:
        if stage in ("train", None):
            self.train_split = torchvision.datasets.MNIST(
                self.datadir, train=True, transform=transforms.ToTensor()
            )
        if stage in ("test", None):
            self.test_split = torchvision.datasets.MNIST(
                self.datadir, train=False, transform=transforms.ToTensor()
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_split,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=min(8, os.cpu_count() or 0),
            drop_last=True,
        )


class LitCIFAR10(pl.LightningDataModule):
    pass


class LitImageNet(pl.LightningDataModule):
    pass
