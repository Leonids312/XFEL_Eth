import copy
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from area_page import AreaAnalysisPage, ImageEntry, Spot
from contour_editor import OutlineDraft
from pathlib import Path


class ContourEditTests(unittest.TestCase):
    bounds = (10, 10, 100, 100)
    corners = [[20, 20], [45, 20], [45, 45], [20, 45]]

    def test_native_polygon_preserved_and_draft_is_private(self):
        native = [self.corners[0], [23,20]] + self.corners[1:]
        result = {"status":"自动", "contour":native}
        unchanged = copy.deepcopy(result)
        draft = OutlineDraft(result, self.bounds)
        self.assertEqual(draft.polygon(), native)
        selected = draft.insert((30, 20), 2)
        self.assertIsNotNone(selected)
        draft.move(selected, (30, 15))
        self.assertGreater(cv2.contourArea(np.array(draft.polygon(), np.int32)), 25**2)
        draft.remove(selected)
        self.assertEqual(result, unchanged)
        self.assertIsNone(draft.insert((0,0), 100))

    def test_review_contour_is_a_proposal_only(self):
        result = {"status":"待复核", "contour":None, "area_px":None, "review_contour":self.corners}
        self.assertEqual(OutlineDraft(result, self.bounds).polygon(), self.corners)
        self.assertIsNone(result["area_px"])

    def test_open_ends_require_two_manual_caps_and_no_self_intersections(self):
        paths = [[[25,20], [25,50]], [[40,50], [40,20]]]
        draft = OutlineDraft({"status":"待复核", "method":"dic", "paths":paths,
                              "review_contour":self.corners}, self.bounds)
        for tip in (None,(32,18)):
            draft.tips[0] = tip
            with self.assertRaisesRegex(ValueError, "两个端部"):
                draft.polygon()
        draft.tips[1] = (32,52)
        self.assertEqual(len(draft.polygon()), 6)
        draft.move((1,0),(18,21))
        with self.assertRaisesRegex(ValueError,"自交"):
            draft.polygon()
        with self.assertRaisesRegex(ValueError,"截断"):
            OutlineDraft({"status":"图像截断","method":"dic","paths":paths}, self.bounds)

    def test_missing_geometry_and_image_boundary_rejected(self):
        with self.assertRaises(ValueError):
            OutlineDraft({}, self.bounds)
        d = OutlineDraft({"contour":[[30,30],[90,20],[101,20],[50,30]]}, self.bounds)
        with self.assertRaises(ValueError):
            d.polygon()

    def test_only_confirm_writes_and_cannot_write_into_another_entry(self):
        page = AreaAnalysisPage.__new__(AreaAnalysisPage)
        spot = Spot("6",1,1,1,1,1,"test")
        entry = ImageEntry(Path("a.tif"),"a.tif",[spot],1,1)
        page.current_entry, page.selected_spot_uid = entry, "test"
        page._analysis_cancel = None
        page.scale_um_per_px = .5
        page.column_order_var = Mock()
        page.load_image = Mock(return_value=np.zeros((120,120,3),np.uint8))
        page._changed_area = Mock()
        page.results = {"test":{"status":"待复核","review_contour":self.corners,"area_px":None}}
        with patch("contour_editor.edit_outline") as window:
            page.edit_selected_contour()
        commit = window.call_args.args[-1]
        self.assertIsNone(page.results["test"]["area_px"])
        page.current_entry = None
        with self.assertRaises(ValueError):
            commit(self.corners)
        page.current_entry = entry
        commit(self.corners)
        self.assertEqual(page.results["test"]["status"], "手动画")
        self.assertAlmostEqual(page.results["test"]["area_um2"], 625/4)
        page._changed_area.assert_called_once()


if __name__ == "__main__":
    unittest.main()
