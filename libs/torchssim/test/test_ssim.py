import unittest
import numpy as np
import torch
from torchimagefilter import ImageFilter, GaussFilter
from torchssim import MSSIM, SimplifiedMSSIM
from utils import make_reproducible, load_reference


class Tester(unittest.TestCase):
    def get_matlab_default_image_filter(self) -> ImageFilter:
        return GaussFilter(
            std=1.5, radius=5, output_shape="same", padding_mode="replicate"
        )

    def test_ssim_cpu(self):
        make_reproducible()

        mssim = MSSIM(image_filter=self.get_matlab_default_image_filter())
        mssim = mssim.double()

        actual_scores = []
        desired_scores = []
        for image1, image2, desired_score in load_reference():
            with torch.no_grad():
                image1, image2 = image1.double(), image2.double()
                actual_score = mssim(image1, image2).item()

            actual_scores.append(actual_score)
            desired_scores.append(desired_score)

        actual = np.array(actual_scores)
        desired = np.array(desired_scores)

        np.testing.assert_allclose(actual, desired, atol=5e-3, rtol=0.0)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_ssim_cuda(self):
        make_reproducible()

        mssim = MSSIM(image_filter=self.get_matlab_default_image_filter())
        mssim = mssim.double().cuda()

        actual_scores = []
        desired_scores = []
        for image1, image2, desired_score in load_reference():
            with torch.no_grad():
                image1, image2 = image1.double().cuda(), image2.double().cuda()
                actual_score = mssim(image1, image2).item()

            actual_scores.append(actual_score)
            desired_scores.append(desired_score)

        actual = np.array(actual_scores)
        desired = np.array(desired_scores)

        np.testing.assert_allclose(actual, desired, atol=5e-3, rtol=0.0)

    def test_simplified_ssim_cpu(self):
        make_reproducible()

        simplified_mssim = SimplifiedMSSIM(
            image_filter=self.get_matlab_default_image_filter()
        )
        simplified_mssim = simplified_mssim.double()

        actual_scores = []
        desired_scores = []
        for image1, image2, desired_score in load_reference():
            with torch.no_grad():
                image1, image2 = image1.double(), image2.double()
                actual_score = simplified_mssim(image1, image2).item()

            actual_scores.append(actual_score)
            desired_scores.append(desired_score)

        actual = np.array(actual_scores)
        desired = np.array(desired_scores)

        np.testing.assert_allclose(actual, desired, atol=5e-3, rtol=0.0)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_simplified_ssim_cuda(self):
        make_reproducible()

        simplified_mssim = SimplifiedMSSIM(
            image_filter=self.get_matlab_default_image_filter()
        )
        simplified_mssim = simplified_mssim.double().cuda()

        actual_scores = []
        desired_scores = []
        for image1, image2, desired_score in load_reference():
            with torch.no_grad():
                image1, image2 = image1.double().cuda(), image2.double().cuda()
                actual_score = simplified_mssim(image1, image2).item()

            actual_scores.append(actual_score)
            desired_scores.append(desired_score)

        actual = np.array(actual_scores)
        desired = np.array(desired_scores)

        np.testing.assert_allclose(actual, desired, atol=5e-3, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
