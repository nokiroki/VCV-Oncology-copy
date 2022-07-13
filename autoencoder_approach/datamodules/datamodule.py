import os
from typing import Optional

import cv2
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
from torchvision import transforms


class MelanomaDataset(Dataset):

    def __init__(self, data_dir: str = 'data/', transform=None) -> None:
        super(MelanomaDataset, self).__init__()
        self.data_dir = data_dir
        self.img_list = os.listdir(data_dir)
        self.transform = transform

    def __getitem__(self, index: int) -> T_co:
        x = cv2.cvtColor(cv2.imread(os.path.join(self.data_dir, self.img_list[index])), cv2.COLOR_BGR2RGB)
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self) -> int:
        return len(self.img_list)


class MelanomaDataModule(LightningDataModule):

    def __init__(self,
                 data_train_dir: str,
                 data_test_dir: str,
                 data_val_dir: str,
                 batch_size: int = 16,
                 num_workers: int = 4,
                 train_transforms: Optional[transforms.Compose] = None,
                 val_test_transforms: Optional[transforms.Compose] = None) -> None:
        super().__init__()

        self.data_train_dir = data_train_dir
        self.data_test_dir = data_test_dir
        self.data_val_dir = data_val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_test_transforms = val_test_transforms

        self.train = None
        self.val = None
        self.test = None

    def prepare_data(self) -> None: ...

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_transforms is None:
            self.train_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=40, scale=(.9, 1.1), shear=0),
                transforms.RandomPerspective(distortion_scale=.2),
                transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5),
                transforms.ToTensor(),
                transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
            ])
        if self.val_test_transforms is None:
            self.val_test_transforms = transforms.Compose((
                transforms.ToTensor(),
                transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
            ))

        if stage in (None, 'fit'):
            self.train = MelanomaDataset(self.data_train_dir, transform=self.train_transforms)
            self.val = MelanomaDataset(self.data_val_dir, transform=self.val_test_transforms)

        elif stage in (None, 'test'):
            self.test = MelanomaDataset(self.data_test_dir, transform=self.val_test_transforms)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
