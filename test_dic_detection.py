import unittest

import cv2
import numpy as np

from dic_detection import detect_dic


def make_stripe(sign=1, *, start=25, stop=190, amplitude=25., single=False):
    y, x = np.mgrid[:220, :90]
    t = np.clip((y-start) / (stop-start), 0., 1.)
    half_width = 8 * np.sin(np.pi*t)**.4
    mid = 45. + .02*(y-110)
    left = np.exp(-.5*((x-mid+half_width)/1.3)**2)
    right = np.exp(-.5*((x-mid-half_width)/1.3)**2)
    band = amplitude * sign * (y >= start) * (y <= stop) * (left-(0 if single else right))
    # Chroma signal that almost cancels under luminance weighting.
    rgb = 100.+x[...,None]*.03+band[...,None]*np.array([2.,-1.,0.])
    noise = np.random.default_rng(23).normal(0, 1, rgb.shape)
    return rgb + noise, half_width[:,0]


class TestDIC(unittest.TestCase):
    def test_chromatic_pair_area_both_bias_polarities(self):
        for sign in (-1, 1):
            image, w = make_stripe(sign)
            im = np.clip(image, 0, 255).astype(np.uint8)
            out = detect_dic(im)
            self.assertEqual(out['status'], '自动', out['diagnostics'])
            area = cv2.contourArea(np.asarray(out['contour'], np.float32))
            expected = 2*w.sum()
            self.assertLess(abs(area-expected)/expected, .06)
            self.assertTrue(np.all(np.ptp(out['contour'], axis=0) > 0))

    def test_keeps_uint16_signal_smaller_than_one_display_level(self):
        image, _ = make_stripe(amplitude=12.)
        im = (image - 100 + 31000).astype(np.uint16)
        self.assertLessEqual(np.ptp(im >> 8), 1)
        out = detect_dic(im)
        self.assertEqual(out['status'], '自动', out['diagnostics'])

    def test_horizontal_and_vertical_coordinates(self):
        image, _ = make_stripe()
        image = image.astype(np.float32)
        a = detect_dic(image)
        b = detect_dic(image.swapaxes(0,1), axis="horizontal")
        self.assertEqual(a['status'], '自动')
        np.testing.assert_array_equal(a['contour'], np.asarray(b['contour'])[:, ::-1])

    def test_truncated_never_closes(self):
        image, _ = make_stripe(start=-20)
        out = detect_dic(image)
        self.assertEqual(out['status'], '图像截断')
        self.assertIsNone(out['contour'])
        self.assertTrue(out['paths'])

    def test_clipped_channels_are_not_required_for_localization(self):
        image, _ = make_stripe()
        im = np.clip(image, 0, 255).astype(np.uint8)
        im[...,0] = 255
        im[...,2] = 0
        out = detect_dic(im)
        self.assertEqual(out['status'], '自动', out['diagnostics'])
        self.assertEqual(out['diagnostics']['channels'], 'G')
        im[...,1] = 255
        out = detect_dic(im)
        self.assertIsNone(out['contour'])
        self.assertIn('截', out['diagnostics']['reason'])

    def test_dust_not_forced_into_a_stripe(self):
        image = np.full((260,110,3), 120, dtype=np.uint8)
        for row,col,r in ((90,24,8),(125,32,9),(160,65,6)):
            cv2.circle(image,(col,row),r,(50,60,70),-1)
            cv2.circle(image,(col+3,row),r//2,(240,220,180),-1)
        out = detect_dic(image)
        self.assertIsNone(out['contour'], out['diagnostics'])

    def test_missing_data_does_not_create_area(self):
        image, _ = make_stripe()
        image[85:125] = 100.
        for a in (image, make_stripe(single=True)[0], np.zeros((150,90),np.uint16),
                  np.random.default_rng(42).normal(100,2,(220,90,3)).astype(np.float32)):
            out = detect_dic(a)
            self.assertIsNone(out['contour'], out['diagnostics'])


if __name__ == '__main__':
    unittest.main()
