import numpy as np
from pytorch_lightning import loggers as pl_loggers, Trainer

from networks import Conv2dAutoEncoder, ConvUp2dAutoEncoder
from datamodules.datamodule import MelanomaDataset, MelanomaDataModule

import torch

import unittest


class TestConvAutoEncoder(unittest.TestCase):

    def setUp(self) -> None:
        self.model = Conv2dAutoEncoder(3, 8)
        self.x = torch.randn(10, 3, 256, 256)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = self.model.to(device)
        self.x = self.x.to(device)

    def test_shapes(self) -> None:
        with torch.no_grad():
            x = torch.clone(self.x)
            print('=' * 20, '\nEncoder shapes:')
            for layer in self.model.encoder:
                x = layer(x)
                print(f'{layer.__class__.__name__} - {x.shape}')
            print('=' * 20)

            print('=' * 20, '\nDecoder shapes:')
            for layer in self.model.decoder:
                x = layer(x)
                print(f'{layer.__class__.__name__} - {x.shape}')
            print('=' * 20)

    def test_output_shape(self) -> None:
        with torch.no_grad():
            img, latent = self.model(self.x)

        print(f'Img shape is - {img.shape}')
        print(f'Latent shape is - {latent.shape}')

    def test_all_good(self) -> None:
        with torch.no_grad():
            out = self.model.predict_step(self.x)

        print(out)


class TestUpConvAutoEncoder(unittest.TestCase):

    def setUp(self) -> None:
        self.model = ConvUp2dAutoEncoder()
        self.x = torch.randn(10, 3, 256, 256)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = self.model.to(device)
        self.x = self.x.to(device)

    def test_shapes(self) -> None:
        with torch.no_grad():
            x = torch.clone(self.x)
            print('=' * 20, '\nEncoder shapes:')
            for layer in self.model.encoder:
                x = layer(x)
                print(f'{layer.__class__.__name__} - {x.shape}')
            print('=' * 20)

            print('=' * 20, '\nDecoder shapes:')
            for layer in self.model.decoder:
                x = layer(x)
                print(f'{layer.__class__.__name__} - {x.shape}')
            print('=' * 20)

    def test_output_shape(self) -> None:
        with torch.no_grad():
            img, latent = self.model(self.x)

        print(f'Img shape is - {img.shape}')
        print(f'Latent shape is - {latent.shape}')

    def test_all_good(self) -> None:
        with torch.no_grad():
            out = self.model.predict_step(self.x)

        print(out)


class TestDataSets(unittest.TestCase):

    def setUp(self) -> None:
        self.train_dataset = MelanomaDataset('unittest_dataset/normal_train')

    def test_len(self) -> None:
        self.assertEqual(len(self.train_dataset), 81)

    def test_object(self) -> None:
        img = self.train_dataset[0]
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape, (256, 256, 3))


class TestDataModules(unittest.TestCase):

    def test_creation(self) -> None:
        data_module = MelanomaDataModule('unittest_dataset/normal_train',
                                         'unittest_dataset/test',
                                         'unittest_dataset/normal_val')
        data_module.setup('fit')
        data_module.setup('test')

    def test_is_none(self) -> None:
        data_module = MelanomaDataModule('unittest_dataset/normal_train',
                                         'unittest_dataset/test',
                                         'unittest_dataset/normal_val')
        data_module.setup('fit')
        with self.subTest():
            self.assertIsNone(data_module.test)

        data_module = MelanomaDataModule('unittest_dataset/normal_train',
                                         'unittest_dataset/test',
                                         'unittest_dataset/normal_val')

        data_module.setup('test')
        with self.subTest():
            self.assertIsNone(data_module.train)
            self.assertIsNone(data_module.val)

    def test_fit(self) -> None:
        data_module = MelanomaDataModule('unittest_dataset/normal_train',
                                         'unittest_dataset/test',
                                         'unittest_dataset/normal_val')
        data_module.setup('fit')

        train = data_module.train
        val = data_module.val

        with self.subTest():
            self.assertEqual(len(train), 81)
            self.assertEqual(len(val), 27)

        train_dataloader = iter(data_module.train_dataloader())
        val_dataloader = iter(data_module.val_dataloader())

        with self.subTest():
            self.assertEqual(next(train_dataloader).shape, (16, 3, 256, 256))
            self.assertEqual(next(val_dataloader).shape, (16, 3, 256, 256))

    def test_test(self) -> None:
        data_module = MelanomaDataModule('unittest_dataset/normal_train',
                                         'unittest_dataset/test',
                                         'unittest_dataset/normal_val')
        data_module.setup('test')

        test = data_module.test

        with self.subTest():
            self.assertEqual(len(test), 27)

        test_dataloader = iter(data_module.test_dataloader())

        with self.subTest():
            self.assertEqual(next(test_dataloader).shape, (16, 3, 256, 256))


class TestLearning(unittest.TestCase):

    def setUp(self) -> None:
        self.model_conv = Conv2dAutoEncoder(3, 8)
        self.model_convup = ConvUp2dAutoEncoder()

        self.logger = pl_loggers.TensorBoardLogger('pl_logs', 'test_logs')
        self.trainer = Trainer(gpus=1, max_epochs=2, logger=self.logger)
        self.dm = MelanomaDataModule('unittest_dataset/normal_train',
                                'unittest_dataset/test',
                                'unittest_dataset/normal_val',
                                batch_size=2)

    def test_training_conv(self) -> None:
        try:
            self.trainer.fit(self.model_conv, self.dm)
            self.trainer.test(self.model_conv, self.dm)
        except RuntimeError:
            print('RunTime Error. Not enough memory for testing, Quitting')

    def test_training_convup(self) -> None:
        try:
            self.trainer.fit(self.model_convup, self.dm)
            self.trainer.test(self.model_convup, self.dm)
        except RuntimeError:
            print('RunTime Error. Not enough memory for testing, Quitting')


if __name__ == '__main__':
    calc_test_suit = unittest.TestSuite()

    calc_test_suit.addTest(unittest.makeSuite(TestConvAutoEncoder))
    calc_test_suit.addTest(unittest.makeSuite(TestUpConvAutoEncoder))
    calc_test_suit.addTest(unittest.makeSuite(TestDataSets))
    calc_test_suit.addTest(unittest.makeSuite(TestDataModules))
    calc_test_suit.addTest(unittest.makeSuite(TestLearning))

    print(f'Amount of tests - {calc_test_suit.countTestCases()}')

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(calc_test_suit)


