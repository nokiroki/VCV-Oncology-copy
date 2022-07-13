import argparse
from typing import Optional
import os

from pytorch_lightning import loggers as pl_loggers, Trainer

from networks import Conv2dAutoEncoder, ConvUp2dAutoEncoder
from datamodules.datamodule import MelanomaDataset, MelanomaDataModule


def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', dest='model', default='cae', type=str)
    parser.add_argument('--dir', dest='dir', default='.\\', type=str)
    parser.add_argument('--batch', dest='batch_size', default=16, type=int)
    parser.add_argument('--epochs', dest='num_epochs', default=20, type=int)
    parser.add_argument('--name', dest='name', default='default', type=str)
    parser.add_argument('--save_dir', dest='save_dir', default=None, type=Optional[str])
    parser.add_argument('--save_every', dest='save_every', default=5, type=int)

    return parser


if __name__ == '__main__':
    parser = configure_parser()
    args = parser.parse_args()

    if args.model == 'caeup':
        model = ConvUp2dAutoEncoder()
    else:
        model = Conv2dAutoEncoder(3, 8)

    logger = pl_loggers.TensorBoardLogger('pl_logs', args.name)
    trainer = Trainer(gpus=1, max_epochs=args.num_epochs, logger=logger)
    dm = MelanomaDataModule(os.path.join(args.dir, 'normal_train'),
                            os.path.join(args.dir, 'test'),
                            os.path.join(args.dir, 'normal_val'),
                            batch_size=args.batch_size)

    trainer.fit(model, dm)
