import time
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from .cae import Conv2dAutoEncoder


class Conv2dAutoEncoderClassifier(pl.LightningModule):

    def __init__(
        self,
        in_channels: int,
        n_latent_features: int,
        cae_backbone: Optional[Conv2dAutoEncoder] = None,
        lr_mlp: float = 1e-3,
        dropout: float = .5
    ):
        super().__init__()
        self.cae = cae_backbone if cae_backbone is not None else Conv2dAutoEncoder(in_channels, n_latent_features)
        self.cae.requires_grad_(False)

        self.mlp = nn.Sequential(
            nn.Linear(25 * 25 * n_latent_features, 2048),
            nn.BatchNorm1d(2048),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(2048, 128),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.lr_mlp = lr_mlp

    def forward(self, x, meta):
        cae_embed = self.cae(x)['latent']
        cae_embed = cae_embed.reshape((cae_embed.shape[0], -1))
        return self.mlp(cae_embed)

    def predict_step(self, x, meta):
        prob = self(x, meta)
        return prob > .5

    def training_step(self, batch, batch_idx):
        x, meta, labels = batch
        out = self.predict_step(x, meta)
        loss = nn.BCELoss()(out, labels)
        f1 = f1_score(labels, out.detach().cpu().numpy())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_f1_score", f1, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": out['loss'], "f1_score": f1}

    def training_epoch_end(self, outputs):
        self.log("train_time", time.time() - self.time_start, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        x, meta, labels = batch
        out = self.predict_step(x, meta)
        loss = nn.BCELoss()(out, labels)
        f1 = f1_score(labels, out.cpu().numpy())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1_score", f1, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "f1_score": f1}

    def validation_epoch_end(self, outputs):
        pass

    def test_step(self, batch, batch_idx: int):
        x, meta, labels = batch
        out = self.predict_step(x, meta)
        loss = nn.BCELoss()(out, labels)
        f1 = f1_score(labels, out.cpu().numpy())
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_f1_score", f1, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "f1_score": f1}

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=self.lr_mlp)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', .5, 2)

        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
            'monitor': 'val_f1_score'
        }
