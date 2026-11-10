import unittest

import cv2
import numpy as np

from pit_detection import detect_pit


class PitContourTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((192, 192, 3), 120, np.uint8)
        self.y, self.x = np.mgrid[:192, :192]
        self.disk = ((self.x - 96) / 37)**2 + ((self.y - 92) / 24)**2 <= 1

    def test_opposite_contrast_and_isoluminant_pit(self):
        # At 8-bit quantization all three regions are exactly equal in the
        # ordinary OpenCV grayscale image, so a luminance threshold cannot work.
        a = self.rgb
        a[self.disk & (self.x < 96)] = (65,148,120)
        a[self.disk & (self.x >= 96)] = (175,92,120)
        self.assertEqual(np.ptp(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)), 0)
        result = detect_pit(a)
        self.assertEqual(result["status"], "自动")
        area = cv2.contourArea(np.array(result["contour"], np.int32))
        self.assertLess(abs(area/self.disk.sum() - 1), .06)

    def test_16bit_low_contrast_not_truncated_to_bytes(self):
        a = np.full(self.rgb.shape, 31000, np.uint16)
        a[self.disk & (self.x < 96)] += 35
        a[self.disk & (self.x >= 96)] -= 36
        result = detect_pit(a)
        self.assertEqual(result["status"], "自动")
        self.assertLess(abs(cv2.contourArea(np.array(result["contour"]))/self.disk.sum()-1), .07)

    def test_long_gap_does_not_measure_an_open_crescent(self):
        cv2.ellipse(self.rgb,(96,92),(37,24),0,0,170,(200,190,160),3)
        cv2.ellipse(self.rgb,(96,92),(37,24),0,180,293,(54,75,82),3)
        result = detect_pit(self.rgb)
        self.assertEqual(result["status"],"待复核")
        self.assertIsNone(result["contour"])
        self.assertTrue(result.get("review_contour"))

    def test_large_roi_returns_full_resolution_polygon(self):
        a = np.full((1200,1270,3),31000,np.uint16)
        cv2.circle(a,(590,650),85,(31042,30996,31011),-1)
        result = detect_pit(a)
        self.assertEqual(result["status"],"自动")
        p = np.asarray(result["contour"])
        self.assertLess(abs(cv2.contourArea(p)/(np.pi*85**2)-1), .04)
        np.testing.assert_allclose((p.max(axis=0)+p.min(axis=0))/2, [590,650],atol=2)

    def test_twin_candidates_are_not_arbitrarily_picked(self):
        for xx in (61,129):
            cv2.circle(self.rgb,(xx,95),16,(196,190,183),-1)
        result = detect_pit(self.rgb)
        self.assertEqual(result["status"],"待复核")
        self.assertIsNone(result["contour"])

    def test_blank_noise_is_not_zero_area_or_a_false_contour(self):
        rng = np.random.default_rng(22)
        for img in (self.rgb, (self.rgb+rng.normal(0,4,self.rgb.shape)).astype(np.uint8),
                    np.full((50,60), 59641, np.uint16)):
            result = detect_pit(img)
            self.assertEqual(result["status"],"未检出")
            self.assertIsNone(result["contour"])

    def test_scratch_alone_is_not_a_pit(self):
        cv2.line(self.rgb,(2,52),(190,134),(35,45,52),2)
        self.assertIsNone(detect_pit(self.rgb)["contour"])

    def test_pit_cut_by_frame_is_not_closed(self):
        cv2.ellipse(self.rgb,(3,92),(30,26),0,0,360,(185,164,174),-1)
        result = detect_pit(self.rgb)
        self.assertEqual(result["status"],"图像截断")
        self.assertIsNone(result["contour"])


if __name__ == "__main__":
    unittest.main()
