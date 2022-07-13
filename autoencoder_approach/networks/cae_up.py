import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    """
    Simple weight initialization
    """
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        m.bias.data.fill_(0.01)


class ConvUp2dAutoEncoder(pl.LightningModule):

    def __init__(
        self,
        in_channels: int = 3,
        n_latent_features: int = 2048,
        dropout: float = .5
    ):
        super().__init__()
        self.out = n_latent_features
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2),
            nn.BatchNorm2d(512),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2),
            nn.BatchNorm2d(1024),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=n_latent_features, kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.encoder.apply(init_weights)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=n_latent_features, out_channels=1024, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2),
            nn.BatchNorm2d(512),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=in_channels, kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.decoder.apply(init_weights)

        self.train_index = 0
        self.val_index = 0
        self.final_labels = None
        self.time_start = time.time()

    def forward(self, x):
        """
        Returns embeddings
        """
        encoded_img = self.encoder(x)
        latent = torch.squeeze(encoded_img)
        return encoded_img, latent

    def predict_step(self, x):
        encoded_img, latent = self(x)
        decoded_img = self.decoder(encoded_img)
        decoded_img = F.interpolate(decoded_img, x.shape[2:], mode='bilinear', align_corners=True)
        loss = torch.nn.MSELoss()(decoded_img, x)
        return {'latent': latent, 'loss': loss}

    def training_step(self, batch, batch_idx):
        x = batch
        out = self.predict_step(x)
        self.log("train_loss", out['loss'], on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": out['loss']}

    def training_epoch_end(self, outputs):
        self.log("train_time", time.time() - self.time_start, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        x = batch
        out = self.predict_step(x)
        self.log("val_loss", out['loss'], on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": out['loss']}

    def validation_epoch_end(self, outputs):
        pass

    def test_step(self, batch, batch_idx: int):
        x = batch
        out = self.predict_step(x)
        self.log("test_loss", out['loss'], on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": out['loss'], "latent": out['latent']}

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=.003)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', .2, 2),
            'monitor': 'val_loss'
        }

        return [optimizer], scheduler
