"""App integration tests using temporary synthetic data, not laboratory files."""

import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from area_detection import crossings, cross_section, measure_image, read_native_rgb, spot_bounds
from area_page import AreaAnalysisPage, ProcessingParameters, read_image_rgb
from xfel_core import ProjectState, PointRecord


class AreaModeIntegrationTests(unittest.TestCase):
    def test_rotated_camera_file_uses_preview_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            pixels = np.zeros((12, 27, 3), np.uint8)
            pixels[:4, :8] = [242, 80, 31]
            image = Image.fromarray(pixels)
            exif = image.getexif()
            exif[274] = 6
            path = Path(directory) / "rotated.jpg"
            image.save(path, exif=exif)
            np.testing.assert_array_equal(read_native_rgb(path), read_image_rgb(path))

    def test_native_16bit_rgb_order_phase_and_shared_boundary(self):
        rgb = np.zeros((20, 30, 3), dtype=np.uint16)
        rgb[:] = [56789, 32101, 1000]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "16位.tif"
            ok, buf = cv2.imencode(".tif", rgb[:, :, ::-1])
            self.assertTrue(ok)
            buf.tofile(str(path))
            np.testing.assert_array_equal(read_native_rgb(path), rgb)
            cells = [("k", 1, 2, 1, 2)]
            def detector(roi, **kwargs):
                np.testing.assert_array_equal(roi, rgb[:, 15:])
                self.assertEqual(kwargs["axis"], "vertical")
                return {"status": "自动", "contour": [[0, 1], [8, 1], [8, 11], [0, 11]],
                        "diagnostics": {"width": np.int32(8)}}
            with patch("dic_detection.detect_dic", side_effect=detector):
                m = measure_image(path, "grazing_columns", cells, "layout", 6, 3)["k"]
            self.assertEqual(m["area_px"], 80)
            self.assertEqual(m["contour"][0], [15, 1])
            self.assertEqual(json.loads(json.dumps(m)), m)

    def test_review_draft_is_not_a_measurement(self):
        cells = [("k", 2, 1, 2, 1)]
        result = {"status": "待复核", "paths": [[[3, 0], [3, 6]]],
                  "review_contour": [[1, 1], [4, 1], [1, 3]], "diagnostics": {"reason": "缺口"}}
        with patch("area_detection.read_native_rgb", return_value=np.zeros((20, 10, 3), np.uint8)), \
             patch("pit_detection.detect_pit", return_value=result):
            m = measure_image("unused", "normal", cells, "layout", 6, 3)["k"]
        self.assertIsNone(m["area_px"])
        self.assertIsNone(m["contour"])
        self.assertEqual(m["paths"], [[[3, 10], [3, 16]]])
        self.assertEqual(m["review_contour"][0], [1, 11])
        self.assertEqual(m["diagnostics"]["reason"], "缺口")

    def test_cancellation_does_not_publish_partial_image(self):
        flag = threading.Event()
        def detect(roi, **kwargs):
            self.assertEqual(kwargs["axis"], "horizontal")
            flag.set()
            return {"status": "图像截断"}
        with patch("area_detection.read_native_rgb", return_value=np.zeros((20, 10, 3), np.uint8)), \
             patch("dic_detection.detect_dic", side_effect=detect) as d:
            r = measure_image("unused", "grazing", [("a",2,1,1,1),("b",2,1,2,1)],
                              "layout", 6, 3, flag.is_set)
        self.assertEqual(r, {})
        self.assertEqual(d.call_count, 1)

    def test_manual_vs_algorithm_truncation(self):
        self.assertTrue(AreaAnalysisPage._locked({"status": "图像截断"}))
        self.assertFalse(AreaAnalysisPage._locked({"status": "图像截断", "method": "dic"}))
        self.assertTrue(AreaAnalysisPage._locked({"status": "手动画"}))
        self.assertEqual(ProcessingParameters().method, "threshold")

    def test_scale_applies_to_inactive_layout_but_not_other_folders(self):
        page = AreaAnalysisPage.__new__(AreaAnalysisPage)
        page.scale_um_per_px = .5
        page.root_dir = Path("sample")
        page.results = {"A": {"area_px": 8}}
        page.inactive_results = {"normal": {"B": {"area_px": 12}, "C": {"area_px": None}}}
        page.state = ProjectState()
        for i, directory in enumerate(("sample", "different"), 1):
            point = PointRecord(atten="2", serial=i, area=72, area_px=12, image_path=str(Path(directory)/"a.tif"))
            page.state.records[point.uid] = point
        page._recompute_scaled_areas()
        self.assertEqual(page.results["A"]["area_um2"], 2)
        self.assertEqual(page.inactive_results["normal"]["B"]["area_um2"], 3)
        self.assertIsNone(page.inactive_results["normal"]["C"]["area_um2"])
        self.assertEqual(page.state.records["2::1"].area, 3)
        self.assertEqual(page.state.records["2::2"].area, 72)

    def test_profile_coordinate_orientation_and_open_paths(self):
        y, x = np.indices((12, 10))
        rgb = np.stack((x, y, x+y), axis=-1)
        at, profile = cross_section(rgb, "vertical", .5)
        np.testing.assert_allclose(profile[:, 0], np.arange(10))
        np.testing.assert_allclose(profile[:, 1], at)
        at, profile = cross_section(rgb, "horizontal", .5)
        np.testing.assert_allclose(profile[:, 1], np.arange(12))
        contour = [[1,1],[7,1],[7,9],[1,9]]
        np.testing.assert_allclose(sorted(crossings(contour, "horizontal", 4)), [1,9])
        np.testing.assert_allclose(crossings(contour, "vertical", 4, closed=False), [7])
        self.assertEqual(spot_bounds(101, 91, 1, 13, 1, 13), (84, 0, 91, 101))


if __name__ == "__main__":
    unittest.main()
