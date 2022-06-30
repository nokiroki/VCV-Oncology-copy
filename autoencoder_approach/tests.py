from networks.cae import Conv2dAutoEncoder

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


if __name__ == '__main__':
    calc_test_suit = unittest.TestSuite()
    calc_test_suit.addTest(unittest.makeSuite(TestConvAutoEncoder))
    print(f'Amount of tests - {calc_test_suit.countTestCases()}')

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(calc_test_suit)


