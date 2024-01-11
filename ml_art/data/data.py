import os
import torch
import omegaconf

from torch.utils.data import DataLoader


def wiki_art(cfg: omegaconf.dictconfig.DictConfig):
    """Return train and test dataloaders for WikiArt."""

    train_loader = DataLoader(
        dataset=torch.load(
            os.path.join(cfg.dataset.processed_path, "train_set.pt")
        ),
        batch_size=cfg.dataset.batch_size,
        shuffle=cfg.dataset.dataloader_shuffle,
    )

    test_loader = DataLoader(
        dataset=torch.load(
            os.path.join(cfg.dataset.processed_path, "test_set.pt")
        ),
        batch_size=cfg.dataset.batch_size,
        shuffle=cfg.dataset.dataloader_shuffle,
    )

    return train_loader, test_loader
