# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications Copyright Skoltech Deep Learning Course.

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from scipy import stats

from .model import Generator, Discriminator
from .loss import GANLoss



class GAN(pl.LightningModule):
    def __init__(
        self,
        loss_type: str,
        class_conditional: bool,
        truncation_trick: bool,
        data_path: str,
        batch_size: int = 64,
        lr_gen: float = 1e-4,
        lr_dis: float = 4e-4
    ):
        super().__init__()
        self.data_path = data_path
        self.noise_channels = 128
        self.class_conditional = class_conditional
        self.truncation_trick = truncation_trick

        self.batch_size = batch_size
        self.lr_gen = lr_gen
        self.lr_dis = lr_dis
        self.eps = 1e-7
        self.num_classes = 17

        self.train_dataset = datasets.ImageFolder(data_path, transform=transforms.ToTensor())

        self.gen = Generator(
            min_channels=32, 
            max_channels=512, 
            noise_channels=self.noise_channels, 
            num_classes=self.num_classes, 
            num_blocks=4,
            use_class_condition=class_conditional)

        self.dis = Discriminator(
            min_channels=32, 
            max_channels=512, 
            num_classes=self.num_classes, 
            num_blocks=4,
            use_projection_head=class_conditional)

        self.loss = GANLoss(loss_type)
        self.loss_type = loss_type

        self.register_buffer('val_noise', torch.randn(len(self.train_dataset), self.noise_channels), persistent=False)

        if self.truncation_trick:
            self.val_noise.data = torch.FloatTensor(stats.truncnorm.rvs(-1, 1, self.val_noise.data.shape))

    def forward(self, noise, labels):
        return self.gen(noise, labels)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, labels = batch

        noise = torch.randn(imgs.shape[0], self.noise_channels).to(imgs.device)
        real_scores = self.dis(imgs, labels)
        fake_scores = self.dis(self.forward(noise, labels), labels)

        if optimizer_idx == 0:
            
            loss = self.loss(fake_scores) if self.loss_type == "hinge" else self.loss(torch.sigmoid(fake_scores))
                
        elif optimizer_idx == 1:
            
            real_scores = self.dis(imgs, labels)
            
            loss = self.loss(fake_scores, real_scores) if self.loss_type == "hinge" else self.loss(torch.sigmoid(fake_scores), torch.sigmoid(real_scores))

        assert loss.shape == (), 'Loss should be a single digit'

        loss_name = 'loss_gen' if optimizer_idx == 0 else 'loss_dis'
        self.log(loss_name, loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        opt_gen = torch.optim.Adam(self.gen.parameters(), lr=self.lr_gen, betas=(0.0, 0.999))
        opt_dis = torch.optim.Adam(self.dis.parameters(), lr=self.lr_dis, betas=(0.0, 0.999))

        return [opt_gen, opt_dis], []

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=8, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=8, batch_size=self.batch_size, shuffle=False)

    @torch.no_grad()
    def on_epoch_end(self):
        # Visualize Images
        noise = self.val_noise[:16 * self.num_classes]
        labels = torch.arange(self.num_classes).repeat_interleave(16, dim=0).to(noise.device)

        fake_imgs = self.forward(noise, labels)
        fake_imgs = fake_imgs.detach().cpu()

        grid = utils.make_grid(fake_imgs, nrow=16)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)
