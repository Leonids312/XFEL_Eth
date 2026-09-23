"""Core data and numerical routines for the integrated XFEL damage app."""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


GMD_KEY = "SBP:FE:GMD1_6485:Curr:AI"
GMD_TS_KEY = "SBP:FE:GMD1_6485:Curr:AI.timestamp"
IMAGE_TS_KEY = "4K_1:image1:TimeStamp_RBV.timestamp"
PD_KEY = "mso:Area"
POSITION_KEY = "position_list"
SHOT_NUMBER_KEY = "shot_number"

COLUMN_NAMES = {
    "A": "serial",
    "B": "gmd",
    "C": "pd",
    "D": "energy",
    "E": "area",
    "F": "energy_rel_err",
    "G": "area_rel_err",
}
NAME_TO_COLUMN = {value: key for key, value in COLUMN_NAMES.items()}


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def attenuation_sort_key(value: str) -> tuple[int, float | str]:
    match = re.search(r"[+-]?\d+(?:\.\d+)?", str(value))
    if match:
        return (0, float(match.group()))
    return (1, str(value).casefold())


@dataclass
class PointRecord:
    atten: str
    serial: int
    gmd: float | None = None
    pd: float | None = None
    energy: float | None = None
    area: float | None = None
    energy_rel_err: float | None = None
    area_rel_err: float | None = None
    image_path: str = ""
    image_rel_path: str = ""
    image_display_rows: int | None = None
    image_display_cols: int | None = None
    spot_display_row: int | None = None
    spot_display_col: int | None = None
    contour: list[list[int]] | None = None
    area_status: str = "未处理"
    area_unit: str = "µm²"
    area_px: float | None = None
    source_hdf5: str = ""
    gmd_shift: int | None = None
    deleted: bool = False
    deleted_at: str = ""
    notes: str = ""

    @property
    def uid(self) -> str:
        return f"{self.atten}::{self.serial}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PointRecord":
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in fields})


class ProjectState:
    """Single source of truth shared by the three pages."""

    def __init__(self) -> None:
        self.records: dict[str, PointRecord] = {}
        self.formulas: dict[str, dict[str, str]] = {}
        self.constants: dict[str, float] = default_constants()
        self.gmd_pd_fit: dict[str, Any] = {}
        self.area_display_style: dict[str, Any] = {}
        self.alignment_log: list[dict[str, Any]] = []
        self._listeners: list[Callable[[str], None]] = []

    def subscribe(self, listener: Callable[[str], None]) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def notify(self, reason: str) -> None:
        for listener in list(self._listeners):
            listener(reason)

    def groups(self) -> list[str]:
        return sorted({record.atten for record in self.records.values()}, key=attenuation_sort_key)

    def group_records(self, atten: str, include_deleted: bool = True) -> list[PointRecord]:
        rows = [record for record in self.records.values() if record.atten == atten]
        if not include_deleted:
            rows = [record for record in rows if not record.deleted]
        return sorted(rows, key=lambda record: record.serial)

    def ensure_rows(self, atten: str, start: int, end: int) -> list[PointRecord]:
        atten = str(atten).strip()
        if not atten:
            raise ValueError("衰减倍数不能为空。")
        if start <= 0 or end < start:
            raise ValueError("序号范围无效，应满足 1 ≤ 起始序号 ≤ 结束序号。")
        rows = []
        for serial in range(start, end + 1):
            uid = f"{atten}::{serial}"
            if uid not in self.records:
                self.records[uid] = PointRecord(atten=atten, serial=serial)
            rows.append(self.records[uid])
        self.notify("rows")
        return rows

    def merge_area_records(self, records: Iterable[dict[str, Any]], unit: str) -> None:
        touched: set[str] = set()
        for item in records:
            atten = str(item["atten"])
            serial = int(item["serial"])
            uid = f"{atten}::{serial}"
            record = self.records.get(uid) or PointRecord(atten=atten, serial=serial)
            record.area = finite_float(item.get("area"))
            record.area_px = finite_float(item.get("area_px"))
            record.area_unit = unit
            record.area_status = str(item.get("status", "未处理"))
            record.image_path = str(item.get("image_path", ""))
            record.image_rel_path = str(item.get("image_rel_path", ""))
            record.image_display_rows = int(item["image_display_rows"]) if item.get("image_display_rows") is not None else None
            record.image_display_cols = int(item["image_display_cols"]) if item.get("image_display_cols") is not None else None
            record.spot_display_row = int(item["spot_display_row"]) if item.get("spot_display_row") is not None else None
            record.spot_display_col = int(item["spot_display_col"]) if item.get("spot_display_col") is not None else None
            record.contour = item.get("contour")
            self.records[uid] = record
            touched.add(uid)
        self.notify("area")

    def delete_point(self, uid: str) -> None:
        record = self.records[uid]
        record.deleted = True
        record.deleted_at = datetime.now().isoformat(timespec="seconds")
        self.notify("deleted")

    def restore_point(self, uid: str) -> None:
        record = self.records[uid]
        record.deleted = False
        record.deleted_at = ""
        self.notify("restored")

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [record.to_dict() for record in self.records.values()],
            "formulas": self.formulas,
            "constants": self.constants,
            "gmd_pd_fit": self.gmd_pd_fit,
            "area_display_style": self.area_display_style,
            "alignment_log": self.alignment_log,
        }

    def load_dict(self, data: dict[str, Any], notify: bool = True) -> None:
        self.records = {}
        for item in data.get("records", []):
            record = PointRecord.from_dict(item)
            self.records[record.uid] = record
        self.formulas = {
            str(group): {str(column): str(formula) for column, formula in formulas.items()}
            for group, formulas in data.get("formulas", {}).items()
        }
        # Project files created before the normalized-slope constant existed
        # stored the same quantity as Bfit/exp(k*Pprime) inside formulas. Keep
        # old constants available internally, but migrate formulas to the new
        # no-background model on load.
        for formulas in self.formulas.values():
            for column, formula in list(formulas.items()):
                formula = re.sub(
                    r"Bfit\s*/\s*exp\(\s*k\s*\*\s*Pprime\s*\)",
                    "normalized_slope",
                    formula,
                )
                formula = formula.replace(
                    "(sigBfit/Bfit)^2", "(sig_normalized_slope/normalized_slope)^2",
                )
                formula = re.sub(
                    r"\(\s*P\s*\*\s*sigk\s*\)\s*\^\s*2",
                    "((P-Pprime)*sigk)^2",
                    formula,
                )
                if column == "pd":
                    # Neither PD nor GMD background participates in the
                    # current central-value model. Migrate common legacy
                    # placements of the additive PD intercept.
                    formula = re.sub(r"\s*\+\s*intercept\b", "", formula)
                    formula = re.sub(r"\bintercept\s*\+\s*", "", formula)
                if column == "energy_rel_err" and any(
                    token in formula for token in ("intercept", "sigPDBackground", "sigBG", "IBG")
                ):
                    formula = (
                        "=sqrt(gmd_rel_err^2+"
                        "(sig_normalized_slope/normalized_slope)^2+"
                        "((P-Pprime)*sigk)^2+(sigeta/eta)^2)"
                    )
                formulas[column] = formula
        constants = default_constants()
        stored_constants = {key: float(value) for key, value in data.get("constants", {}).items()}
        constants.update(stored_constants)
        if "normalized_slope" not in stored_constants:
            try:
                scale = math.exp(float(constants["k"]) * float(constants["Pprime"]))
                constants["normalized_slope"] = float(constants["Bfit"]) / scale
            except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
                pass
        if "sig_normalized_slope" not in stored_constants:
            try:
                scale = abs(math.exp(float(constants["k"]) * float(constants["Pprime"])))
                constants["sig_normalized_slope"] = float(constants["sigBfit"]) / scale
            except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
                pass
        self.constants = constants
        self.gmd_pd_fit = dict(data.get("gmd_pd_fit", {}))
        self.area_display_style = dict(data.get("area_display_style", {}))
        self.alignment_log = list(data.get("alignment_log", []))
        if notify:
            self.notify("load")


def default_constants() -> dict[str, float]:
    bfit = 2.578
    coefficient = -41.62
    calibration_pressure = 0.11
    normalized_slope = bfit / math.exp(coefficient * calibration_pressure)
    return {
        # Bfit/Pprime/sigBfit are retained as hidden compatibility constants
        # for old custom formulas.  New UI and formulas expose the normalized
        # slope as one quantity.
        "Bfit": bfit,
        # Background-related names remain at zero only so old custom project
        # files can still be parsed. They are hidden and unused by defaults.
        "intercept": 0.0,
        "sigBfit": 0.0,
        "normalized_slope": normalized_slope,
        "sig_normalized_slope": 0.0,
        "sigPDBackground": 0.0,
        "k": coefficient,
        "sigk": 0.0,
        "P": 0.0,
        "Pprime": calibration_pressure,
        "load_resistance": 50.0,
        "responsivity": 103.91,
        "eta": 0.27,
        "sigeta": 0.004,
        "wavelength_A": 31.0,
        "q": 1.602176634e-19,
        "gmd_rel_err": 0.10,
        "IBG": 0.0,
        "sigBG": 0.0,
        "area_rel_default": 0.0,
    }


class FormulaError(ValueError):
    pass


ALLOWED_FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "ln": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "min": min,
    "max": max,
    "pow": pow,
}


class SafeFormulaEvaluator(ast.NodeVisitor):
    def __init__(self, variables: dict[str, float]) -> None:
        self.variables = variables

    def evaluate(self, expression: str) -> float:
        expression = normalize_formula(expression)
        try:
            tree = ast.parse(expression, mode="eval")
            result = self.visit(tree.body)
            number = float(result)
        except FormulaError:
            raise
        except Exception as exc:
            raise FormulaError(str(exc)) from exc
        if not math.isfinite(number):
            raise FormulaError("计算结果不是有限数。")
        return number

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise FormulaError("公式只允许数值常量。")

    def visit_Name(self, node: ast.Name) -> float:
        if node.id in self.variables:
            return float(self.variables[node.id])
        if node.id == "pi":
            return math.pi
        if node.id == "e":
            return math.e
        raise FormulaError(f"未知变量：{node.id}")

    def visit_BinOp(self, node: ast.BinOp) -> float:
        left = self.visit(node.left)
        right = self.visit(node.right)
        operators = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.Pow: lambda a, b: a**b,
            ast.Mod: lambda a, b: a % b,
        }
        function = operators.get(type(node.op))
        if function is None:
            raise FormulaError("公式包含不支持的运算符。")
        return function(left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        value = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise FormulaError("公式包含不支持的一元运算符。")

    def visit_Call(self, node: ast.Call) -> float:
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
            raise FormulaError("公式包含不支持的函数。")
        if node.keywords:
            raise FormulaError("函数不支持关键字参数。")
        return float(ALLOWED_FUNCTIONS[node.func.id](*(self.visit(arg) for arg in node.args)))

    def generic_visit(self, node: ast.AST) -> Any:
        raise FormulaError(f"公式包含不允许的语法：{type(node).__name__}")


def normalize_formula(formula: str) -> str:
    expression = formula.strip()
    if expression.startswith("="):
        expression = expression[1:]
    expression = re.sub(r"col\(\s*([A-Ga-g])\s*\)", lambda m: m.group(1).upper(), expression, flags=re.I)
    expression = re.sub(r"\b([A-Ga-g])\$?\d+\b", lambda m: m.group(1).upper(), expression)
    expression = expression.replace("^", "**")
    return expression


def row_variables(record: PointRecord, constants: dict[str, float]) -> dict[str, float]:
    variables = {key: float(value) for key, value in constants.items()}
    for letter, name in COLUMN_NAMES.items():
        value = getattr(record, name)
        if value is not None:
            variables[letter] = float(value)
            variables[letter.lower()] = float(value)
            variables[name] = float(value)
    try:
        variables["atten"] = float(record.atten)
    except ValueError:
        pass
    return variables


def apply_formula(
    records: Iterable[PointRecord],
    target: str,
    formula: str,
    constants: dict[str, float],
) -> list[tuple[str, str]]:
    if target not in NAME_TO_COLUMN:
        raise FormulaError(f"不支持给列 {target} 设置公式。")
    errors: list[tuple[str, str]] = []
    for record in records:
        try:
            value = SafeFormulaEvaluator(row_variables(record, constants)).evaluate(formula)
            setattr(record, target, value)
        except FormulaError as exc:
            setattr(record, target, None)
            errors.append((record.uid, str(exc)))
    return errors


def nearest_indices(reference_timestamps: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference_timestamps, dtype=float).reshape(-1)
    target = np.asarray(target_timestamps, dtype=float).reshape(-1)
    if reference.size == 0:
        raise ValueError("GMD 时间戳为空。")
    if np.all(np.diff(reference) >= 0):
        right = np.clip(np.searchsorted(reference, target), 0, reference.size - 1)
        left = np.clip(right - 1, 0, reference.size - 1)
        choose_left = np.abs(reference[left] - target) <= np.abs(reference[right] - target)
        return np.where(choose_left, left, right).astype(int)
    return np.asarray([int(np.argmin(np.abs(reference - value))) for value in target], dtype=int)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    x, y = x[valid], y[valid]
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    averages = ends - (counts - 1) / 2.0
    return averages[inverse]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    return _pearson(_ranks(x[valid]), _ranks(y[valid]))


def _require_h5_dataset(handle: Any, key: str) -> np.ndarray:
    if key not in handle:
        raise KeyError(f"HDF5 文件缺少数据集：{key}")
    return np.asarray(handle[key])


@dataclass
class AlignmentResult:
    values: np.ndarray
    shift: int
    pearson: float
    spearman: float
    score: float
    level: str
    candidate_count: int
    shot_number: int
    source_count: int
    mean_time_difference: float


def align_damage_gmd(
    file_path: str | Path,
    measured_area: Iterable[float | None],
    expected_count: int,
    shift_min: int = -10,
    shift_max: int = 10,
) -> AlignmentResult:
    """Align camera shutter timestamps to GMD and select the strongest positive trend."""
    import h5py

    areas = np.asarray([np.nan if value is None else float(value) for value in measured_area], dtype=float)
    if len(areas) != expected_count:
        raise ValueError(f"面积数据有 {len(areas)} 项，但所选序号需要 {expected_count} 项。")
    with h5py.File(file_path, "r") as handle:
        gmd = _require_h5_dataset(handle, GMD_KEY).astype(float).reshape(-1)
        gmd_ts = _require_h5_dataset(handle, GMD_TS_KEY).astype(float).reshape(-1)
        camera_ts = _require_h5_dataset(handle, IMAGE_TS_KEY).astype(float).reshape(-1)
        positions = np.asarray(handle[POSITION_KEY]) if POSITION_KEY in handle else np.empty((0, 2))
        if SHOT_NUMBER_KEY in handle:
            raw_shots = float(np.asarray(handle[SHOT_NUMBER_KEY]).reshape(-1)[0])
            shot_number = int(round(raw_shots))
            if shot_number <= 0 or not math.isclose(raw_shots, shot_number):
                raise ValueError(f"shot_number 无效：{raw_shots}")
        else:
            shot_number = 1
    if len(gmd) != len(gmd_ts):
        raise ValueError(f"GMD 数值数目 {len(gmd)} 与时间戳数目 {len(gmd_ts)} 不一致。")
    base = nearest_indices(gmd_ts, camera_ts)
    if expected_count == len(camera_ts):
        level = "per shutter"
    elif len(positions) and expected_count == len(positions) and len(camera_ts) == len(positions) * shot_number:
        level = "per position mean"
    else:
        position_hint = len(positions) if len(positions) else "未知"
        raise ValueError(
            f"数量不匹配：所选序号 {expected_count} 个；HDF5 有 {len(camera_ts)} 个快门时间戳、"
            f"{position_hint} 个位置，shot_number={shot_number}。"
        )

    candidates: list[dict[str, Any]] = []
    for shift in range(int(shift_min), int(shift_max) + 1):
        indices = base + shift
        if np.any(indices < 0) or np.any(indices >= len(gmd)):
            continue
        raw = gmd[indices]
        values = raw if level == "per shutter" else raw.reshape(len(positions), shot_number).mean(axis=1)
        valid = np.isfinite(values) & np.isfinite(areas) & (values > 0)
        pearson = _pearson(np.log(values[valid]), areas[valid]) if np.count_nonzero(valid) >= 2 else float("nan")
        spearman = _spearman(values[valid], areas[valid]) if np.count_nonzero(valid) >= 2 else float("nan")
        score = 0.5 * pearson + 0.5 * spearman if np.isfinite(pearson) and np.isfinite(spearman) else -np.inf
        candidates.append(
            {
                "values": values,
                "shift": shift,
                "pearson": pearson,
                "spearman": spearman,
                "score": score,
                "mean_dt": float(np.mean(gmd_ts[indices] - camera_ts)),
            }
        )
    if not candidates:
        raise ValueError(f"偏移范围 [{shift_min}, {shift_max}] 内没有索引全部有效的方案。")
    if all(not np.isfinite(item["score"]) for item in candidates):
        raise ValueError("面积数据有效变化点少于 2 个，无法按正相关度选择时间戳偏移。")
    best = max(
        candidates,
        key=lambda item: (
            item["score"],
            item["pearson"] if np.isfinite(item["pearson"]) else -np.inf,
            item["spearman"] if np.isfinite(item["spearman"]) else -np.inf,
            -abs(item["shift"]),
        ),
    )
    return AlignmentResult(
        values=np.asarray(best["values"], dtype=float),
        shift=int(best["shift"]),
        pearson=float(best["pearson"]),
        spearman=float(best["spearman"]),
        score=float(best["score"]),
        level=level,
        candidate_count=len(candidates),
        shot_number=shot_number,
        source_count=len(camera_ts),
        mean_time_difference=float(best["mean_dt"]),
    )


@dataclass
class GmdPdFit:
    offset: int
    slope: float
    intercept: float
    r: float
    r_squared: float
    slope_stderr: float
    n: int
    gmd: np.ndarray = field(repr=False)
    pd: np.ndarray = field(repr=False)
    all_results: list[dict[str, float]] = field(default_factory=list)
    curves: list[dict[str, Any]] = field(default_factory=list, repr=False)


def fit_gmd_pd(file_path: str | Path, offsets: Iterable[int]) -> GmdPdFit:
    import h5py

    with h5py.File(file_path, "r") as handle:
        gmd = _require_h5_dataset(handle, GMD_KEY).astype(float).reshape(-1)
        pd_raw = _require_h5_dataset(handle, PD_KEY).astype(float).reshape(-1)
    results = []
    for offset in dict.fromkeys(int(value) for value in offsets):
        n = min(len(gmd), len(pd_raw))
        if n <= abs(offset):
            continue
        if offset >= 0:
            x, y = gmd[offset:n], -pd_raw[: n - offset]
        else:
            x, y = gmd[: n + offset], -pd_raw[-offset:n]
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if len(x) < 3 or np.ptp(x) == 0:
            continue
        design = np.column_stack([x, np.ones_like(x)])
        params, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        if rank < 2:
            continue
        slope, intercept = map(float, params)
        fitted = design @ params
        residual = y - fitted
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
        r = float(np.corrcoef(x, y)[0, 1])
        dof = len(x) - 2
        stderr = math.sqrt((ss_res / dof) / float(np.sum((x - np.mean(x)) ** 2))) if dof > 0 else float("nan")
        results.append(
            {"offset": offset, "slope": slope, "intercept": intercept, "r": r,
             "r_squared": r2, "slope_stderr": stderr, "n": len(x), "gmd": x, "pd": y}
        )
    if not results:
        raise ValueError("没有可拟合的 GMD–PD 数据，请检查文件和偏移范围。")
    best = max(results, key=lambda item: item["r_squared"] if np.isfinite(item["r_squared"]) else -np.inf)
    summary = [{key: float(value) for key, value in item.items() if key not in {"gmd", "pd"}} for item in results]
    curves = [
        {
            **{key: float(value) for key, value in item.items() if key not in {"gmd", "pd"}},
            "gmd": np.asarray(item["gmd"], dtype=float),
            "pd": np.asarray(item["pd"], dtype=float),
        }
        for item in results
    ]
    return GmdPdFit(
        offset=int(best["offset"]), slope=float(best["slope"]), intercept=float(best["intercept"]),
        r=float(best["r"]), r_squared=float(best["r_squared"]), slope_stderr=float(best["slope_stderr"]),
        n=int(best["n"]), gmd=np.asarray(best["gmd"]), pd=np.asarray(best["pd"]), all_results=summary, curves=curves,
    )


@dataclass
class ThresholdFit:
    a: float
    b: float
    a_err: float
    b_err: float
    r_squared: float
    threshold: float
    threshold_err: float
    used_mask: np.ndarray
    sigma: float
    iterations: int
    threshold_fit_err: float = float("nan")
    threshold_x_err: float = 0.0
    threshold_y_err: float = 0.0


def _log_fit(energy: np.ndarray, area: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.log(np.asarray(energy, dtype=float))
    y = np.asarray(area, dtype=float)
    design = np.column_stack([x, np.ones_like(x)])
    params, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < 2:
        raise ValueError("ln(E) 缺少足够变化，无法拟合。")
    residual = y - design @ params
    if len(y) > 2:
        sigma2 = float(np.sum(residual**2) / (len(y) - 2))
        try:
            covariance = sigma2 * np.linalg.inv(design.T @ design)
        except np.linalg.LinAlgError:
            covariance = np.full((2, 2), np.nan)
    else:
        covariance = np.full((2, 2), np.nan)
    return params, covariance, residual


def fit_threshold(
    energy: Iterable[float],
    area: Iterable[float],
    sigma_clip: float | None = None,
    max_iterations: int = 3,
    *,
    energy_rel_err: Iterable[float] | None = None,
    area_rel_err: Iterable[float] | None = None,
) -> ThresholdFit:
    energy_array = np.asarray(list(energy), dtype=float)
    area_array = np.asarray(list(area), dtype=float)
    if len(energy_array) < 2 or len(area_array) != len(energy_array):
        raise ValueError("可用于阈值拟合的数据点少于 2 个。")
    if not np.all(np.isfinite(energy_array)) or not np.all(np.isfinite(area_array)):
        raise ValueError("单脉冲能量和损伤面积必须为有限数值。")
    if np.any(energy_array <= 0):
        raise ValueError("单脉冲能量必须大于 0。")

    def relative_uncertainties(values: Iterable[float] | None, label: str) -> np.ndarray:
        if values is None:
            return np.zeros(len(energy_array), dtype=float)
        result = np.asarray(list(values), dtype=float)
        if result.shape != energy_array.shape:
            raise ValueError(f"{label}数量必须与拟合数据点数量一致。")
        if not np.all(np.isfinite(result)) or np.any(result < 0):
            raise ValueError(f"{label}必须为有限的非负数。")
        return result

    energy_uncertainties = relative_uncertainties(energy_rel_err, "单脉冲能量相对误差")
    area_uncertainties = relative_uncertainties(area_rel_err, "损伤面积相对误差")
    used = np.ones(len(energy_array), dtype=bool)
    iterations = 0
    loop_count = max_iterations if sigma_clip is not None else 1
    for index in range(max(1, loop_count)):
        if np.count_nonzero(used) < 2:
            raise ValueError("异常值剔除后剩余数据点少于 2 个。")
        params, _, residual_used = _log_fit(energy_array[used], area_array[used])
        sigma = float(np.std(residual_used, ddof=2 if len(residual_used) > 2 else 0))
        iterations = index + 1
        if sigma_clip is None or not np.isfinite(sigma) or sigma == 0:
            break
        residual_all = area_array - (params[0] * np.log(energy_array) + params[1])
        new_used = np.abs(residual_all) <= float(sigma_clip) * sigma
        if np.array_equal(new_used, used):
            break
        used = new_used
    params, covariance, residual = _log_fit(energy_array[used], area_array[used])
    a, b = map(float, params)
    if abs(a) <= np.finfo(float).eps:
        raise ValueError("拟合斜率 a 接近 0，无法由 exp(-b/a) 计算阈值。")
    fitted = a * np.log(energy_array[used]) + b
    ss_res = float(np.sum((area_array[used] - fitted) ** 2))
    ss_tot = float(np.sum((area_array[used] - np.mean(area_array[used])) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    errors = np.sqrt(np.diag(covariance)) if np.all(np.isfinite(covariance)) else np.array([np.nan, np.nan])
    threshold = float(np.exp(-b / a))
    gradient = np.array([threshold * b / (a**2), -threshold / a])
    if np.all(np.isfinite(covariance)):
        variance = float(gradient @ covariance @ gradient)
        threshold_fit_err = math.sqrt(variance) if variance >= 0 else float("nan")
    else:
        threshold_fit_err = float("nan")

    # The plotted X uncertainties belong to E, while the regression variable
    # is ln(E), so sigma_ln(E) = sigma_E / E.  Differentiate the least-squares
    # normal equations instead of treating the visible error bars as cosmetic.
    # For X = [ln(E), 1] and residual r = y - X beta:
    #   d beta / d y_i = (X'X)^-1 X_i'
    #   d beta / d x_i = (X'X)^-1 ([r_i, 0]' - a X_i').
    # Applying the threshold gradient to those Jacobians propagates each
    # independent X/Y error bar directly to E_th, even with nonzero residuals.
    fit_energy = energy_array[used]
    fit_area = area_array[used]
    log_energy = np.log(fit_energy)
    design = np.column_stack([log_energy, np.ones_like(log_energy)])
    information_inverse = np.linalg.inv(design.T @ design)
    y_jacobian = information_inverse @ design.T
    x_jacobian = information_inverse @ np.vstack(
        [residual - a * log_energy, np.full(len(log_energy), -a)],
    )
    sigma_log_energy = energy_uncertainties[used]
    sigma_area = np.abs(fit_area) * area_uncertainties[used]
    threshold_x_err = float(np.linalg.norm((gradient @ x_jacobian) * sigma_log_energy))
    threshold_y_err = float(np.linalg.norm((gradient @ y_jacobian) * sigma_area))
    threshold_err = (
        math.hypot(threshold_fit_err, threshold_x_err, threshold_y_err)
        if math.isfinite(threshold_fit_err)
        else float("nan")
    )
    sigma = float(np.std(residual, ddof=2 if len(residual) > 2 else 0))
    return ThresholdFit(
        a, b, float(errors[0]), float(errors[1]), r2, threshold, threshold_err,
        used, sigma, iterations, threshold_fit_err, threshold_x_err, threshold_y_err,
    )


def parse_offsets(text: str) -> list[int]:
    normalized = text.replace("，", ",").replace("；", ",").replace(";", ",").replace("＋", "+").replace("－", "-")
    values: list[int] = []
    for token in re.split(r"[,\s]+", normalized.strip()):
        if not token:
            continue
        if token.startswith("±"):
            number = int(token[1:])
            candidates = [number, -number] if number else [0]
        elif token.startswith("+/-"):
            number = int(token[3:])
            candidates = [number, -number] if number else [0]
        else:
            candidates = [int(token)]
        for value in candidates:
            if value not in values:
                values.append(value)
    if not values:
        raise ValueError("请至少输入一个偏移量。")
    return values


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class RecentPathStore:
    """Persist an independent last-used directory for every file-dialog purpose."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.directories: dict[str, str] = {}
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self.directories = {str(key): str(value) for key, value in raw.get("directories", {}).items()}
            except Exception:
                self.directories = {}

    def initial_dir(self, key: str, fallback: str | Path | None = None) -> str:
        saved = self.directories.get(key, "")
        if saved and Path(saved).is_dir():
            return saved
        candidate = Path(fallback) if fallback else Path.cwd()
        if candidate.is_file():
            candidate = candidate.parent
        return str(candidate if candidate.is_dir() else Path.cwd())

    def remember_file(self, key: str, file_path: str | Path) -> None:
        path = Path(file_path)
        self._remember(key, path.parent)

    def remember_directory(self, key: str, directory: str | Path) -> None:
        self._remember(key, Path(directory))

    def _remember(self, key: str, directory: Path) -> None:
        self.directories[str(key)] = str(directory)
        try:
            self.path.write_text(
                json.dumps({"version": 1, "directories": self.directories}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass
