"""Geometry and pulse-identity regression tests (no external images or display required)."""

import json
import unittest
from pathlib import Path
from unittest.mock import Mock

from area_page import AreaAnalysisPage, compact_layout_hint, mapped_image_entry, parse_image_name
from xfel_core import ProjectState


def page_for(*entries, column_order="LTR", directions=None):
    page = AreaAnalysisPage.__new__(AreaAnalysisPage)
    page.entries = list(entries)
    page.column_order_var = Mock()
    page.column_order_var.get.return_value = column_order
    page.directions = directions or {}
    page.results = {}
    page.state = ProjectState()
    page.scale_um_per_px = None
    page._recompute_serials()
    return page


class GrazingLayoutTests(unittest.TestCase):
    root = Path("sample_images")

    def test_left_right_cuts_and_direction_are_independent_per_group(self):
        a, b = (parse_image_name(self.root / f"atten{att}_no_11-13.tif", self.root,
                                 segmentation_mode="grazing_columns") for att in (3, 6))
        page = page_for(a, b, column_order="RTL", directions={"grazing_columns|3": "RTL", "grazing|6": "BTT"})
        self.assertEqual((a.display_rows, a.display_cols), (1, 3))
        self.assertEqual([page.spot_display_col(a, s) for s in a.spots], [1, 2, 3])
        self.assertEqual([s.serial for s in a.spots], [13, 12, 11])
        self.assertEqual([s.serial for s in b.spots], [11, 12, 13])
        page.publish_areas()
        record = page.state.records["3::13"]
        self.assertEqual((record.spot_display_row, record.spot_display_col), (1, 1))
        self.assertEqual((record.image_display_rows, record.image_display_cols), (1, 3))

    def test_older_grazing_archive_keeps_uid_and_btt(self):
        a = parse_image_name(self.root / "atten1.5_no_21-23.tif", self.root, segmentation_mode="grazing")
        page_for(a, directions={"grazing|1.5": "BTT"})
        self.assertEqual((a.display_rows, a.display_cols), (3, 1))
        self.assertEqual([s.serial for s in a.spots], [23, 22, 21])
        self.assertEqual(a.spots[0].uid, "atten1.5_no_21-23.tif#1:1.5:1:21")

    def test_compact_name_without_row_count_requires_confirmation(self):
        path = self.root / "crystal-atten_6&9-num1-5.tif"
        entry = parse_image_name(path, self.root)
        self.assertEqual((entry.display_rows, entry.display_cols, entry.publish), (4, 5, False))
        page = page_for(entry, directions={"9|2": "RTL"})
        self.assertEqual([s.serial for s in entry.spots[-5:]], [10, 9, 8, 7, 6])
        page.publish_areas()
        self.assertFalse(page.state.records)
        mapping = compact_layout_hint(path.stem)
        mapping.update(publish=True, start=21)
        verified = mapped_image_entry(path, self.root, "normal", mapping)
        pg = page_for(verified, directions={"9|2": "RTL"})
        pg.publish_areas()
        self.assertEqual(set(pg.state.groups()), {"6", "9"})
        self.assertEqual(verified.spots[-1].serial, 26)
        self.assertEqual(verified.spots[0].uid, entry.spots[0].uid)

    def test_unlabelled_bias_images_remain_preview_only(self):
        self.assertIsNone(parse_image_name(self.root / "bias_b.tif", self.root, segmentation_mode="grazing_columns"))
        spec = {"attens": "", "start": 1, "count": 13, "rows": 1, "publish": False}
        one, two = (mapped_image_entry(self.root / f"bias_{label}.tif", self.root, "grazing_columns",
                                       json.loads(json.dumps(spec))) for label in ("a", "b"))
        page = page_for(one, two)
        self.assertEqual((one.display_rows, one.display_cols), (1, 13))
        page.publish_areas()
        self.assertFalse(page.state.records)
        with self.assertRaises(ValueError):
            mapped_image_entry(one.path, self.root, one.layout, dict(spec, publish=True))
        verified = dict(spec, attens="3", publish=True)
        first = mapped_image_entry(one.path, self.root, one.layout, verified)
        second = mapped_image_entry(two.path, self.root, two.layout, verified)
        page = page_for(first, two)
        page.publish_areas()
        with self.assertRaisesRegex(ValueError, "另一张图"):
            page._set_manual_mapping(second, verified)
        self.assertEqual(len(page.state.records), 13)

    def test_invalid_mapping_rejected(self):
        good = dict(attens="1.5", start=1, count=13, rows=1, publish=True)
        for bad in (dict(count=0), dict(start=-1), dict(count=6000000), dict(attens="1.5,3"), dict(attens="未指定")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                mapped_image_entry(self.root / "sample.tif", self.root, "grazing_columns", good | bad)

    def test_reversing_global_r_keeps_manual_outline_on_same_physical_spot(self):
        e = parse_image_name(self.root / "atten3_L1_R1-2.tif", self.root)
        page = page_for(e)
        first, second = e.spots
        page.results[first.uid] = {"status":"手动画", "area_px":10, "contour":[[3,1],[7,1],[5,6]]}
        page.publish_areas()
        page.current_entry = e
        page.selected_spot_uid = first.uid
        page.segmentation_mode_var = Mock()
        page.segmentation_mode_var.get.return_value = "normal"
        page.inactive_results = {}
        page.refresh_spot_tree = page.stop_analysis = page.render_canvas = page.set_status = Mock()
        page.column_order_var.get.return_value = "RTL"
        page.on_column_order_changed()
        self.assertNotIn(first.uid,page.results)
        self.assertIn(second.uid,page.results)
        self.assertEqual(page.selected_spot_uid,second.uid)
        record = page.state.records["3::1"]
        self.assertEqual((record.area,record.spot_display_col,record.contour[0]),(10,1,[3,1]))
        self.assertIsNone(page.state.records["3::2"].area)

    def test_trying_another_grid_preserves_accepted_area_and_its_metadata(self):
        path = self.root / "atten6_no_1-2.tif"
        rows,cols = (parse_image_name(path,self.root,segmentation_mode=m) for m in ("grazing","grazing_columns"))
        page=page_for(rows)
        page.results[rows.spots[0].uid]={"status":"手动画","area_px":12,"contour":[[2,0],[8,0],[2,4]]}
        page.publish_areas()
        page.entries=[cols]; page.results={}; page._recompute_serials()
        page.publish_areas()
        saved=page.state.records["6::1"]
        self.assertEqual((saved.area,saved.image_display_rows,saved.image_display_cols),(12,2,1))
        page.results[cols.spots[0].uid]={"status":"待复核","method":"dic","area_px":None}
        page.publish_areas()
        self.assertEqual(saved.area,12)
        page.results[cols.spots[0].uid]={"status":"自动","area_px":15,"contour":[[2,0],[7,0],[2,6]]}
        page.publish_areas()
        self.assertEqual((saved.area,saved.image_display_rows,saved.image_display_cols),(15,1,2))

    def test_partial_measurement_direction_flip_does_not_double_count(self):
        e = parse_image_name(self.root/"atten3_no_11-12.tif",self.root,segmentation_mode="grazing_columns")
        page=page_for(e)
        page.results[e.spots[0].uid]={"status":"手动画","area_px":12}
        page.publish_areas()
        page.directions["grazing_columns|3"]="RTL"
        page._recompute_serials(); page.publish_areas()
        self.assertEqual([r.area for r in page.state.group_records("3")],[None,12])

    def test_other_source_cannot_silently_replace_same_pulse(self):
        e = parse_image_name(self.root / "atten6_no_1-2.tif", self.root, segmentation_mode="grazing")
        page=page_for(e)
        page.results[e.spots[0].uid]={"status":"自动","area_px":12}
        page.publish_areas()
        other = parse_image_name(self.root/"retake_atten6_no_1-2.tif", self.root, segmentation_mode="grazing")
        page.entries=[other]; page._recompute_serials()
        page.results = {other.spots[0].uid:{"status":"自动","area_px":99}}
        page.publish_areas()
        self.assertEqual(page.state.records["6::1"].area,12)
        self.assertIn(("6",1),page._sync_conflicts)

    def test_deliberate_number_reversal_across_two_images_is_not_a_source_collision(self):
        left, right = [parse_image_name(self.root/f"atten3_L1_R{part}.tif",self.root)
                       for part in ("1-2","3-4")]
        page=page_for(left,right)
        page.segmentation_mode_var=Mock()
        page.segmentation_mode_var.get.return_value="normal"
        for v,s in enumerate(left.spots + right.spots,1):
            page.results[s.uid]={"status":"手动画","area_px":float(v)}
        page.publish_areas()
        page.state.records["3::1"].energy=123
        page.refresh_spot_tree=page.refresh_directions=page.redraw_overlay=page.set_status=Mock()
        page.directions["3|1"]="RTL"
        page._directions_changed(["3|1"])
        self.assertEqual([r.area for r in page.state.group_records("3")],[4,3,2,1])
        self.assertEqual(page.state.records["3::1"].energy,123)
        self.assertFalse(page._sync_conflicts)


if __name__ == "__main__":
    unittest.main()
