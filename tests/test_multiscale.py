import unittest

import torch

from utils import resize_for_scale


class ResizeForScaleTests(unittest.TestCase):
    def test_each_scale_is_applied_to_the_original_image(self):
        image = torch.zeros(1, 3, 100, 50)

        smaller = resize_for_scale(image, 0.9)
        larger = resize_for_scale(image, 1.1)

        self.assertEqual(tuple(smaller.shape), (1, 3, 90, 45))
        self.assertEqual(tuple(larger.shape), (1, 3, 110, 55))
        self.assertEqual(tuple(image.shape), (1, 3, 100, 50))


if __name__ == '__main__':
    unittest.main()
