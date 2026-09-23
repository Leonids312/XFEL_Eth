from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path

import h5py
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from area_page import mapped_grazing_serial, mapped_serial, parse_image_name
from calibration_page import GMDPDCalibrationPage, read_pd_minimum, read_physical_pd
from energy_page import DEFAULT_FORMULAS, EnergyExtractionPage
from threshold_page import ThresholdAnalysisPage
from xfel_core import (
    PointRecord,
    ProjectState,
    RecentPathStore,
    SafeFormulaEvaluator,
    align_damage_gmd,
    apply_formula,
    fit_gmd_pd,
    fit_threshold,
)


class XFELCoreTests(unittest.TestCase):
    def test_extract_and_merge_step2_project_cache(self):
        duplicate = {"cache_id": "cal_1", "source": "run1/data.hdf5", "created_at": "2026-01-01", "best_offset": 0}
        new_item = {"cache_id": "cal_2", "source": "run2/data.hdf5", "created_at": "2026-01-02", "best_offset": 1}
        project = {
            "state": {},
            "calibration_page": {
                "offsets": "0, ±1", "pd_baseline": "2.5",
                "calibration_cache": [duplicate, new_item],
            },
        }
        incoming = GMDPDCalibrationPage.calibration_config_from_project(project)
        current = {"offsets": "0", "calibration_cache": [dict(duplicate)]}
        merged, added, skipped = GMDPDCalibrationPage.merge_calibration_configs(current, incoming)
        self.assertEqual((added, skipped), (1, 1))
        self.assertEqual([item["cache_id"] for item in merged["calibration_cache"]], ["cal_1", "cal_2"])
        self.assertEqual(merged["offsets"], "0, ±1")
        self.assertEqual(merged["pd_baseline"], "2.5")

    def test_extract_step2_cache_from_legacy_energy_page(self):
        project = {"energy_page": {"calibration_cache": [{"cache_id": "old"}], "offsets": "0, 1"}}
        extracted = GMDPDCalibrationPage.calibration_config_from_project(project)
        self.assertEqual(extracted["calibration_cache"][0]["cache_id"], "old")
        self.assertEqual(extracted["offsets"], "0, 1")

    def test_threshold_attenuation_filter_selects_only_requested_groups(self):
        records = [
            PointRecord(atten="3", serial=1, energy=1.0, area=2.0),
            PointRecord(atten="6", serial=1, energy=2.0, area=3.0),
            PointRecord(atten="9", serial=1, energy=3.0, area=4.0),
            PointRecord(atten="3", serial=2, energy=4.0, area=5.0),
        ]
        mask = ThresholdAnalysisPage._attenuation_selection_mask(records, {"3", "9"})
        np.testing.assert_array_equal(mask, [True, False, True, True])

    def test_origin_excel_and_named_formula_syntax(self):
        variables = {"A": 2.0, "B": 3.0, "gain": 4.0}
        self.assertAlmostEqual(SafeFormulaEvaluator(variables).evaluate("=col(A)^2+B2*gain"), 16.0)

    def test_formula_column_calculation(self):
        state = ProjectState()
        rows = state.ensure_rows("3", 1, 3)
        for index, record in enumerate(rows, 1):
            record.gmd = index * 1e-8
        errors = apply_formula(rows, "pd", "=Bfit*gmd", state.constants)
        self.assertEqual(errors, [])
        self.assertTrue(all(record.pd is not None for record in rows))

    def test_normalized_slope_formula_and_legacy_project_migration(self):
        state = ProjectState()
        state.load_dict({
            "formulas": {"3": {
                "pd": "=Bfit/exp(k*Pprime)*exp(k*P)*gmd+intercept",
                "energy_rel_err": "=sqrt(gmd_rel_err^2+(P*sigk)^2+(sigPDBackground/pd)^2)",
            }},
            "constants": {"Bfit": 4.0, "Pprime": 0.2, "k": -2.0, "intercept": 0.5},
        }, notify=False)
        expected = 4.0 / np.exp(-2.0 * 0.2)
        self.assertAlmostEqual(state.constants["normalized_slope"], expected)
        self.assertIn("normalized_slope", state.formulas["3"]["pd"])
        self.assertNotIn("Bfit/exp", state.formulas["3"]["pd"])
        self.assertNotIn("intercept", state.formulas["3"]["pd"])
        self.assertIn("(P-Pprime)*sigk", state.formulas["3"]["energy_rel_err"])
        self.assertNotIn("sigPDBackground", state.formulas["3"]["energy_rel_err"])
        self.assertNotIn("sigBG", state.formulas["3"]["energy_rel_err"])
        rows = state.ensure_rows("3", 1, 1)
        rows[0].gmd = 2.0
        errors = apply_formula(rows, "pd", state.formulas["3"]["pd"], state.constants)
        self.assertEqual(errors, [])
        self.assertAlmostEqual(rows[0].pd, expected * 2.0)

    def test_area_preview_metadata_and_style_round_trip(self):
        state = ProjectState()
        state.merge_area_records(
            [{
                "atten": "3", "serial": 7, "area": 12.5, "area_px": 50,
                "status": "自动", "image_path": "image.tif",
                "image_display_rows": 4, "image_display_cols": 5,
                "spot_display_row": 1, "spot_display_col": 3,
                "contour": [[10, 20], [30, 40], [15, 45]],
            }],
            "µm²",
        )
        state.area_display_style = {"selected_color": "#ff8800", "label_font_size": 13}
        restored = ProjectState()
        restored.load_dict(state.to_dict(), notify=False)
        record = restored.records["3::7"]
        self.assertEqual((record.image_display_rows, record.image_display_cols), (4, 5))
        self.assertEqual((record.spot_display_row, record.spot_display_col), (1, 3))
        self.assertEqual(restored.area_display_style["selected_color"], "#ff8800")

    def test_truncated_grazing_stripe_stays_blank_instead_of_zero(self):
        state = ProjectState()
        state.merge_area_records(
            [{
                "atten": "0", "serial": 4, "area": None, "area_px": None,
                "status": "图像截断", "image_path": "atten0_no1-10.tif",
                "contour": None,
            }],
            "µm²",
        )
        record = state.records["0::4"]
        self.assertEqual(record.area_status, "图像截断")
        self.assertIsNone(record.area)
        self.assertIsNone(record.area_px)
        self.assertIsNone(record.contour)

    def test_calibration_cache_source_shows_file_and_two_parents(self):
        label = GMDPDCalibrationPage.source_label(str(Path("C:/experiment/2026/run/data.hdf5")))
        self.assertEqual(Path(label).parts, ("2026", "run", "data.hdf5"))

    def test_calibration_paths_use_natural_numeric_order(self):
        paths = ["2026-05-26/run10/data.hdf5", "2026-05-26/run9/data.hdf5", "2026-05-26/run2/data.hdf5"]
        ordered = sorted(paths, key=GMDPDCalibrationPage.natural_path_key)
        self.assertEqual(ordered, [
            "2026-05-26/run2/data.hdf5", "2026-05-26/run9/data.hdf5", "2026-05-26/run10/data.hdf5",
        ])

    def test_nearest_calibration_slope_uses_hdf5_modification_time(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            damage = root / "damage.hdf5"
            earlier = root / "cal_earlier.hdf5"
            nearest = root / "cal_nearest.hdf5"
            ignored = root / "cal_without_slope.hdf5"
            for path in (damage, earlier, nearest, ignored):
                path.touch()
            os.utime(damage, (1_000_000, 1_000_000))
            os.utime(earlier, (999_000, 999_000))
            os.utime(nearest, (999_990, 999_990))
            os.utime(ignored, (999_999, 999_999))
            match = EnergyExtractionPage.nearest_normalized_slope(damage, [
                {"source": str(earlier), "normalized_slope": 11.0, "normalized_slope_stderr": 0.3, "pprime": 0.05},
                {"source": str(nearest), "normalized_slope": 22.0, "normalized_slope_stderr": 0.4, "pprime": 0.08},
                {"source": str(ignored), "normalized_slope": None, "normalized_slope_stderr": 0.1, "pprime": 0.09},
            ])
            self.assertEqual(match["item"]["source"], str(nearest))
            self.assertEqual(match["slope"], 22.0)
            self.assertEqual(match["slope_stderr"], 0.4)
            self.assertEqual(match["pprime"], 0.08)
            self.assertAlmostEqual(match["time_difference"], 10.0, places=3)

    def test_calibration_range_refit_overwrites_statistics(self):
        x = np.arange(1.0, 8.0)
        y = 3.5 * x + 2.0
        result = GMDPDCalibrationPage._linear_fit(x[2:6], y[2:6], offset=-1)
        self.assertEqual(result["offset"], -1)
        self.assertEqual(result["n"], 4)
        self.assertAlmostEqual(result["slope"], 3.5)
        self.assertAlmostEqual(result["intercept"], 2.0)
        self.assertAlmostEqual(result["r_squared"], 1.0)

    def test_gas_attenuation_fit_reports_k_a_and_sigma_k(self):
        pressures = np.array([0.02, 0.05, 0.10, 0.12, 0.08, 0.06])
        slopes = np.array([0.791, 0.227, 0.0298, 0.012, 0.0657, 0.153])
        result = GMDPDCalibrationPage.fit_gas_attenuation_data(pressures, slopes)
        self.assertAlmostEqual(result["k"], -41.6234505, places=6)
        self.assertAlmostEqual(result["A"], 1.836, places=3)
        self.assertAlmostEqual(result["sigma_k"], 0.358075, places=5)

    def test_pd_background_uncertainty_uses_sem_and_baseline_uncertainty(self):
        result = GMDPDCalibrationPage.pd_background_statistics(
            np.array([1.0, 3.0, 5.0]), baseline=0.5, baseline_sigma=0.2,
        )
        self.assertAlmostEqual(result["background"], 2.5)
        self.assertAlmostEqual(result["sample_std"], 2.0)
        self.assertAlmostEqual(result["mean_sem"], 2.0 / np.sqrt(3.0))
        self.assertAlmostEqual(result["background_sigma"], np.sqrt(4.0 / 3.0 + 0.04))

    def test_energy_error_formula_uses_no_pd_or_gmd_background(self):
        state = ProjectState()
        row = state.ensure_rows("3", 1, 1)[0]
        row.gmd = 2.0e-9
        row.pd = 4.0e-7
        state.constants["P"] = 0.10
        state.constants["Pprime"] = 0.08
        state.constants["sigk"] = 0.2
        state.constants["normalized_slope"] = 5.0
        state.constants["sig_normalized_slope"] = 0.1
        state.constants["intercept"] = 1.0e-8
        state.constants["sigPDBackground"] = 2.0e-9
        errors = apply_formula([row], "energy_rel_err", DEFAULT_FORMULAS["energy_rel_err"], state.constants)
        self.assertEqual(errors, [])
        expected = np.sqrt(0.10**2 + (0.1 / 5.0)**2 + ((0.10 - 0.08) * 0.2)**2 + (0.004 / 0.27)**2)
        self.assertAlmostEqual(row.energy_rel_err, expected)
        self.assertIn("(P-Pprime)*sigk", DEFAULT_FORMULAS["energy_rel_err"])
        self.assertNotIn("intercept", DEFAULT_FORMULAS["pd"])
        self.assertNotIn("intercept", DEFAULT_FORMULAS["energy_rel_err"])
        self.assertNotIn("sigPDBackground", DEFAULT_FORMULAS["energy_rel_err"])
        self.assertNotIn("sigBG", DEFAULT_FORMULAS["energy_rel_err"])

    def test_physical_pd_minimum_uses_negative_mso_area(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pd.hdf5"
            with h5py.File(path, "w") as handle:
                handle["mso:Area"] = [-5.0, -2.0, np.nan, -8.0]
            minimum, count = read_pd_minimum(path)
            self.assertEqual(minimum, 2.0)
            self.assertEqual(count, 3)

    def test_read_physical_pd_returns_finite_sign_corrected_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pd.hdf5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("mso:Area", data=[-1.0, np.nan, -3.5])
            values = read_physical_pd(path)
            np.testing.assert_allclose(values, [1.0, 3.5])

    def test_threshold_fit(self):
        energy = np.array([1.0, 2.0, 4.0, 8.0])
        area = 5.0 * np.log(energy)
        result = fit_threshold(energy, area)
        self.assertAlmostEqual(result.threshold, 1.0, places=10)
        self.assertAlmostEqual(result.r_squared, 1.0, places=10)

    def test_threshold_total_uncertainty_combines_x_y_and_fit(self):
        energy = np.exp(np.array([0.0, 0.6, 1.3, 1.9, 2.6, 3.1]))
        area = 3.5 * np.log(energy) - 0.8 + np.array([0.12, -0.08, 0.04, -0.11, 0.07, -0.03])
        result = fit_threshold(
            energy,
            area,
            energy_rel_err=np.array([0.04, 0.03, 0.05, 0.04, 0.03, 0.05]),
            area_rel_err=np.array([0.06, 0.05, 0.04, 0.06, 0.05, 0.04]),
        )
        self.assertGreater(result.threshold_fit_err, 0)
        self.assertGreater(result.threshold_x_err, 0)
        self.assertGreater(result.threshold_y_err, 0)
        self.assertAlmostEqual(
            result.threshold_err,
            np.sqrt(result.threshold_fit_err**2 + result.threshold_x_err**2 + result.threshold_y_err**2),
            places=12,
        )
        self.assertGreater(result.threshold_err, result.threshold_fit_err)
        text = ThresholdAnalysisPage._format_fit_result(result, 6, 6, 6)
        self.assertIn("E_th =", text)
        self.assertIn("±", text)
        self.assertIn("总误差分量：X=", text)

    def test_threshold_x_y_uncertainties_match_numerical_propagation(self):
        energy = np.exp(np.array([0.1, 0.7, 1.4, 2.0, 2.8]))
        area = 2.8 * np.log(energy) - 0.5 + np.array([0.03, -0.06, 0.08, -0.02, -0.03])
        energy_relative = np.array([0.015, 0.020, 0.018, 0.011, 0.016])
        area_relative = np.array([0.030, 0.025, 0.020, 0.035, 0.028])
        result = fit_threshold(
            energy, area,
            energy_rel_err=energy_relative,
            area_rel_err=area_relative,
        )

        step = 1.0e-5
        x_derivatives = []
        y_derivatives = []
        for index in range(len(energy)):
            energy_plus, energy_minus = energy.copy(), energy.copy()
            energy_plus[index] *= np.exp(step)
            energy_minus[index] *= np.exp(-step)
            x_derivatives.append(
                (fit_threshold(energy_plus, area).threshold - fit_threshold(energy_minus, area).threshold)
                / (2 * step)
            )

            area_plus, area_minus = area.copy(), area.copy()
            area_plus[index] += step
            area_minus[index] -= step
            y_derivatives.append(
                (fit_threshold(energy, area_plus).threshold - fit_threshold(energy, area_minus).threshold)
                / (2 * step)
            )

        expected_x = np.linalg.norm(np.asarray(x_derivatives) * energy_relative)
        expected_y = np.linalg.norm(np.asarray(y_derivatives) * np.abs(area) * area_relative)
        self.assertAlmostEqual(result.threshold_x_err, expected_x, places=9)
        self.assertAlmostEqual(result.threshold_y_err, expected_y, places=9)

    def test_threshold_legacy_call_still_uses_existing_fit_error(self):
        energy = np.array([1.0, 2.0, 4.0, 8.0, 12.0])
        area = 2.0 * np.log(energy) + np.array([0.10, -0.05, 0.06, -0.03, 0.02])
        result = fit_threshold(energy, area)
        self.assertEqual(result.threshold_x_err, 0.0)
        self.assertEqual(result.threshold_y_err, 0.0)
        self.assertAlmostEqual(result.threshold_err, result.threshold_fit_err, places=12)

    def test_threshold_uncertainty_excludes_sigma_rejected_points(self):
        energy = np.exp(np.linspace(0, 2, 10))
        area = 3 * np.log(energy) + np.array([
            0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.02, 0.02, 8.0,
        ])
        energy_errors = np.full(10, 0.10)
        area_errors = np.full(10, 0.05)
        energy_errors[-1] = 1000.0
        area_errors[-1] = 1000.0
        clipped = fit_threshold(
            energy, area, sigma_clip=2.0, max_iterations=5,
            energy_rel_err=energy_errors, area_rel_err=area_errors,
        )
        self.assertFalse(clipped.used_mask[-1])
        expected = fit_threshold(
            energy[clipped.used_mask], area[clipped.used_mask],
            energy_rel_err=energy_errors[clipped.used_mask],
            area_rel_err=area_errors[clipped.used_mask],
        )
        self.assertAlmostEqual(clipped.threshold_x_err, expected.threshold_x_err, places=12)
        self.assertAlmostEqual(clipped.threshold_y_err, expected.threshold_y_err, places=12)
        self.assertAlmostEqual(clipped.threshold_err, expected.threshold_err, places=12)

    def test_threshold_rejects_invalid_relative_uncertainties(self):
        energy = [1.0, 2.0, 4.0]
        area = [0.0, 2.0, 4.0]
        with self.assertRaisesRegex(ValueError, "数量"):
            fit_threshold(energy, area, energy_rel_err=[0.1, 0.2])
        with self.assertRaisesRegex(ValueError, "非负"):
            fit_threshold(energy, area, area_rel_err=[0.1, -0.2, 0.1])

    def test_cached_figure_dpi_does_not_accumulate(self):
        figure = Figure(figsize=(10.2, 7.2), dpi=100)
        ThresholdAnalysisPage._restore_logical_figure_dpi(figure)
        figure._set_dpi(150, forward=False)  # Simulate a 150% Windows display.
        ThresholdAnalysisPage._restore_logical_figure_dpi(figure)
        self.assertEqual(figure.dpi, 100)
        self.assertEqual(figure._original_dpi, 100)

    def test_cached_figure_view_round_trip_preserves_custom_ticks(self):
        figure = Figure()
        axis = figure.add_subplot(111)
        axis.set_xscale("log")
        axis.set_xlim(1.0, 100.0)
        axis.set_ylim(0.0, 50.0)
        axis.yaxis.set_major_locator(MultipleLocator(5.0))
        original = ThresholdAnalysisPage._capture_figure_view(figure)
        axis.set_xlim(2.0, 20.0)
        axis.set_ylim(10.0, 30.0)
        ThresholdAnalysisPage._restore_figure_view(figure, original)
        np.testing.assert_allclose(axis.get_xlim(), (1.0, 100.0))
        np.testing.assert_allclose(axis.get_ylim(), (0.0, 50.0))
        self.assertIsInstance(axis.yaxis.get_major_locator(), MultipleLocator)

    def test_each_dialog_purpose_has_an_independent_persisted_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "calibration"
            second = root / "damage"
            third = root / "single" / "data.hdf5"
            first.mkdir()
            second.mkdir()
            third.parent.mkdir()
            third.touch()
            config = root / "paths.json"
            store = RecentPathStore(config)
            store.remember_directory("energy_calibration_folder_open", first)
            store.remember_directory("energy_damage_hdf5_open", second)
            store.remember_file("calibration_single_hdf5_open", third)
            loaded = RecentPathStore(config)
            self.assertEqual(Path(loaded.initial_dir("energy_calibration_folder_open")), first)
            self.assertEqual(Path(loaded.initial_dir("energy_damage_hdf5_open")), second)
            self.assertEqual(Path(loaded.initial_dir("calibration_single_hdf5_open")), third.parent)

    def test_leica_filename_formats(self):
        root = Path("C:/images")
        full = parse_image_name(root / "atten0,3_L1-2_R1-5.tif", root)
        simple = parse_image_name(root / "atten1.5_no_11-20.tif", root)
        grazing = parse_image_name(
            root / "atten1.5_no_11-20.tif", root, segmentation_mode="grazing",
        )
        self.assertIsNotNone(full)
        self.assertEqual(len(full.spots), 20)
        self.assertEqual({spot.atten for spot in full.spots}, {"0", "3"})
        self.assertIsNotNone(simple)
        self.assertEqual([spot.col for spot in simple.spots], list(range(11, 21)))
        self.assertIsNotNone(grazing)
        self.assertEqual((grazing.display_rows, grazing.display_cols), (10, 1))
        self.assertEqual(grazing.layout, "grazing")
        self.assertEqual([spot.display_row for spot in grazing.spots], list(range(1, 11)))
        self.assertEqual([spot.display_col for spot in grazing.spots], [1] * 10)

    def test_grazing_incidence_vertical_numbering(self):
        top_to_bottom = [mapped_grazing_serial(serial, 11, 20, "TTB") for serial in range(11, 21)]
        bottom_to_top = [mapped_grazing_serial(serial, 11, 20, "BTT") for serial in range(11, 21)]
        self.assertEqual(top_to_bottom, list(range(11, 21)))
        self.assertEqual(bottom_to_top, list(range(20, 10, -1)))

    def test_original_serpentine_direction_numbering(self):
        # Existing archive: global R grows right-to-left; row 1 is RTL and row 2 remains default LTR.
        self.assertEqual(mapped_serial(1, 1, 30, "RTL", "RTL"), 1)
        self.assertEqual(mapped_serial(1, 30, 30, "RTL", "RTL"), 30)
        self.assertEqual(mapped_serial(2, 1, 30, "RTL", "LTR"), 60)
        self.assertEqual(mapped_serial(2, 30, 30, "RTL", "LTR"), 31)

    def test_hdf5_alignment_and_gmd_pd_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.hdf5"
            gmd_ts = np.arange(30, dtype=float)
            gmd = np.random.default_rng(20260720).lognormal(mean=0.0, sigma=0.7, size=30) * 1e-9
            camera_ts = np.arange(5, 15, 2, dtype=float)
            true_indices = camera_ts.astype(int) + 2
            measured = 7 * np.log(gmd[true_indices] * 1e9) + 3
            pd_raw = -(2.5 * gmd + 1.2e-9)
            with h5py.File(path, "w") as handle:
                handle["SBP:FE:GMD1_6485:Curr:AI"] = gmd
                handle["SBP:FE:GMD1_6485:Curr:AI.timestamp"] = gmd_ts
                handle["4K_1:image1:TimeStamp_RBV.timestamp"] = camera_ts
                handle["mso:Area"] = pd_raw
                handle["position_list"] = np.column_stack([np.arange(5), np.arange(5)])
                handle["shot_number"] = [1.0]
            aligned = align_damage_gmd(path, measured, 5, -3, 3)
            self.assertEqual(aligned.shift, 2)
            np.testing.assert_allclose(aligned.values, gmd[true_indices])
            fitted = fit_gmd_pd(path, [-1, 0, 1])
            self.assertEqual(fitted.offset, 0)
            self.assertAlmostEqual(fitted.slope, 2.5, places=10)
            self.assertAlmostEqual(fitted.intercept, 1.2e-9, places=16)


if __name__ == "__main__":
    unittest.main()
