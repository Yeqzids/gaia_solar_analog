#!/usr/bin/env python3
"""
benchmark_name_list_against_catalog.py

Resolve a plain-text list of solar-analog names via SIMBAD, extract Gaia DR3
source IDs, benchmark them against a final Gaia solar-analog catalog, and
optionally plot their spectra against the solar reference.

Input name list format
----------------------
Plain text, one name per line, e.g.

HD 3384
BD-07 435
HD 217014
18 Sco
SA 115-271

Blank lines and lines starting with # are ignored.

Inputs
------
--catalog       Final ranked catalog CSV from your Gaia pipeline
--names         Plain-text file with one literature star name per line
--outdir        Output directory

Optional inputs for plotting / XP access
----------------------------------------
--solar-csv     Solar reference CSV with columns wavelength_nm, flux
--xpdir         Directory containing cached xp_batch_*.parquet files
--wavelength-range LO HI
                Recompute WRMS/SAM/corr/slope_absdiff/feature_penalty using only
                wavelengths within [LO, HI] nm (masks the rest; requires --solar-csv and --xpdir).
                Spanning the full stage2 grid (usually 400–1000 nm end to end) uses the same
                slope_between(450→900 nm) logic as the catalog; a narrower range uses a twoband slope.
                Example: --wavelength-range 550 900 drops λ outside that band from weighted metrics.
--slope-equivalent-percent PC
                Slope-absdiff cutoff as a percent-equivalent (default 1 = same as nominal catalog).
--plot          Make validation plots
--fetch-missing-lit
                Fetch Gaia XP spectra for literature stars not found in cache
                (requires gaiaxpy)

Outputs
-------
1. resolved_name_list.csv
2. literature_name_benchmark_vs_catalog.csv
3. (optional) plot_literature_vs_solar.png and .pdf (spectra normalized to 550 nm, x-axis 400–1000 nm)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astroquery.simbad import Simbad

import gaia_stage2_aip as stage2

try:
    from gaiaxpy import calibrate
except Exception:
    calibrate = None


# ----------------------------------------------------------------------
# Slope-only thresholds (continuum Δλ calibration; must match gaia_stage2_aip.py)
# ----------------------------------------------------------------------

CONTINUUM_WINDOWS_NM = stage2.CONTINUUM_WINDOWS_NM

SLOPE_THRESHOLDS = [
    ("passes_slope_le_0p01", 0.01),
]


# ----------------------------------------------------------------------
# Spectral defaults (must match gaia_stage2_aip scoring grid)
# ----------------------------------------------------------------------

WMIN = float(stage2.DEFAULT_WAVE_MIN)
WMAX = float(stage2.DEFAULT_WAVE_MAX)
WSTEP = float(stage2.DEFAULT_WAVE_STEP)


# ----------------------------------------------------------------------
# General helpers
# ----------------------------------------------------------------------

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def safe_source_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def build_wave_grid(wmin=WMIN, wmax=WMAX, step=WSTEP):
    n = int(np.floor((wmax - wmin) / step)) + 1
    return wmin + step * np.arange(n, dtype=float)


def _nm_grid(wmin: float, wmax: float, step: float) -> np.ndarray:
    """Same grid convention as build_wave_grid for explicit (wmin, wmax)."""
    n = int(np.floor((wmax - wmin) / step)) + 1
    return wmin + step * np.arange(n, dtype=float)


def match_cached_xp_wave_grid(src_n: int, step: float) -> np.ndarray:
    """
    Cached xp_batch flux vectors may be on 330–1050, 400–1050, or the current
    stage2 default grid. Match by length before interpolating to the requested sampling.
    """
    pairs: list[tuple[float, float]] = [
        (330.0, 1050.0),
        (400.0, 1050.0),
        (330.0, 1000.0),
        (400.0, 1000.0),
        (WMIN, WMAX),
    ]
    seen: set[tuple[float, float]] = set()
    for wmin, wmax in pairs:
        key = (wmin, wmax)
        if key in seen:
            continue
        seen.add(key)
        g = _nm_grid(wmin, wmax, step)
        if len(g) == src_n:
            return g
    for anchor in (330.0, 400.0):
        wmax = anchor + (src_n - 1) * step
        if not (900.0 <= wmax <= 1060.0):
            continue
        g = _nm_grid(anchor, wmax, step)
        if len(g) == src_n:
            return g
    return _nm_grid(0.0, (src_n - 1) * step, step)


def nearest_index(arr, value):
    return int(np.nanargmin(np.abs(arr - value)))


def normalize_at_reference_wavelength(
    wave, flux, ref_nm: float, window_nm: float = 5.0
) -> np.ndarray:
    mask = np.isfinite(wave) & np.isfinite(flux) & (np.abs(wave - ref_nm) <= window_nm)
    if mask.sum() < 3:
        idx = nearest_index(wave, ref_nm)
        norm = flux[idx]
    else:
        norm = np.nanmedian(flux[mask])
    if not np.isfinite(norm) or abs(norm) < 1e-20:
        return np.full_like(flux, np.nan)
    return flux / norm


def normalize_at_550nm(wave, flux, window_nm=5.0):
    return normalize_at_reference_wavelength(wave, flux, 550.0, window_nm)


def continuum_window_centers(windows):
    return np.array([(lo + hi) / 2.0 for lo, hi in windows], dtype=float)


def slope_threshold_cut_from_percent(percent: float, windows=CONTINUUM_WINDOWS_NM) -> float:
    centers = continuum_window_centers(windows)
    delta_lambda = float(np.max(centers) - np.min(centers))
    return float(percent / delta_lambda)


def summarize_thresholds(name: str, slope_fraction: float, slope_cut: float) -> str:
    return f"{name}: slope_absdiff <= {slope_cut:.6e} ({slope_fraction * 100.0:g}% slope-equivalent)"


def passes_thresholds(df: pd.DataFrame, slope_absdiff_max: float) -> pd.Series:
    return df["slope_absdiff"].notna() & (df["slope_absdiff"] <= slope_absdiff_max)


def annotate_slope_equivalent_gate(benchmark: pd.DataFrame, slope_fraction: float) -> None:
    """
    Set passes_slope_le_0p01, quality_class from slope_absdiff at the requested slope-equivalent
    (fraction is the same unit as classify_catalog uses, e.g. 0.01 for 1 %).
    """
    benchmark["slope_equivalent_percent"] = float(slope_fraction * 100.0)
    if "slope_absdiff" not in benchmark.columns:
        return
    cut = slope_threshold_cut_from_percent(slope_fraction)
    benchmark["passes_slope_le_0p01"] = passes_thresholds(benchmark, cut)
    benchmark["quality_class"] = np.where(benchmark["passes_slope_le_0p01"], "slope_le_0p01", "other")


def classify_catalog(cat: pd.DataFrame) -> pd.DataFrame:
    out = cat.copy()
    slope_cuts = {
        col: slope_threshold_cut_from_percent(percent)
        for col, percent in SLOPE_THRESHOLDS
    }
    for col, _percent in SLOPE_THRESHOLDS:
        out[col] = passes_thresholds(out, slope_cuts[col])

    out["quality_class"] = np.where(out["passes_slope_le_0p01"], "slope_le_0p01", "other")

    return out


def load_name_list(path: str) -> list[str]:
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            names.append(s)
    return names


# ----------------------------------------------------------------------
# SIMBAD name resolution
# ----------------------------------------------------------------------

def make_simbad():
    s = Simbad()
    s.add_votable_fields("ids", "ra(d)", "dec(d)")
    return s


def canonical_name(s: str | None) -> str:
    """
    Normalize identifiers for comparison.
    """
    if s is None:
        return ""
    s = str(s)
    s = s.replace("−", "-")
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s.strip().upper())
    return s


def strip_simbad_prefix(name: str) -> str:
    """
    Remove common SIMBAD prefixes that are not part of the true identifier.
    """
    n = canonical_name(name)

    prefixes = [
        "NAME ",
        "* ",
        "V* ",
        "PM* ",
        "EM* ",
        "IRAS ",
        "GAIA DR3 ",
        "GAIA DR2 ",
    ]

    for p in prefixes:
        if n.startswith(p):
            return n[len(p):].strip()

    return n


def normalize_alias(name: str) -> str:
    n = strip_simbad_prefix(name)
    n = re.sub(r"\s+", " ", n)
    return n.strip()


def split_ids_blob(ids_blob: str | None) -> list[str]:
    if ids_blob is None:
        return []
    text = str(ids_blob)
    return [normalize_alias(x) for x in text.split("|") if str(x).strip()]


def ids_blob_contains_requested_name(ids_blob: str | None, requested_name: str) -> bool:
    """
    Accept a result if the returned alias list contains the requested name
    after normalization. Also allow compact-space comparison.
    """
    req = normalize_alias(requested_name)
    ids = split_ids_blob(ids_blob)

    if req in ids:
        return True

    req_compact = req.replace(" ", "")
    ids_compact = [x.replace(" ", "") for x in ids]
    if req_compact in ids_compact:
        return True

    return False


def extract_gaia_dr3_id_from_ids(ids_text: str | None):
    if ids_text is None:
        return None

    text = str(ids_text)

    patterns = [
        r"Gaia DR3\s+(\d+)",
        r"GAIA DR3\s+(\d+)",
        r"Gaia DR2\s+(\d+)",
        r"GAIA DR2\s+(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return int(m.group(1))
    return None


def candidate_name_variants(name: str) -> list[str]:
    """
    Generate careful name variants while preserving full identifiers.
    """
    raw = re.sub(r"\s+", " ", str(name).strip())
    canon = canonical_name(raw)

    out = [raw, canon]

    # HD nnnnn
    m = re.match(r"^HD\s+(\d+)$", canon)
    if m:
        out.append(f"HD {m.group(1)}")

    # BD+24 2810 / BD-07 435
    m = re.match(r"^(BD[+-]\d+)\s+(\d+)$", canon)
    if m:
        out.append(f"{m.group(1)} {m.group(2)}")
        out.append(f"BD {m.group(1)[2:]} {m.group(2)}")

    # SA 115-271
    m = re.match(r"^(SA)\s+(.+)$", canon)
    if m:
        out.append(f"{m.group(1)} {m.group(2)}")

    out.append(raw.title() if raw.isupper() else raw)

    seen = set()
    uniq = []
    for x in out:
        key = canonical_name(x)
        if key not in seen:
            uniq.append(x)
            seen.add(key)
    return uniq


def resolve_name(simbad: Simbad, name: str, debug: bool = False) -> dict:
    tried = candidate_name_variants(name)

    for cand in tried:
        try:
            ids_tbl = simbad.query_objectids(cand)
        except Exception:
            ids_tbl = None

        ids_blob = None
        if ids_tbl is not None and len(ids_tbl) > 0:
            id_list = []
            for row in ids_tbl:
                val = row["ID"]
                if isinstance(val, bytes):
                    val = val.decode()
                id_list.append(str(val))
            ids_blob = "|".join(id_list)

        if ids_blob is not None and not ids_blob_contains_requested_name(ids_blob, cand):
            if debug:
                print(f"Rejected alias mismatch: input={name!r}, cand={cand!r}, ids={ids_blob}")
            ids_blob = None

        if ids_blob is not None:
            try:
                obj_tbl = simbad.query_object(cand)
            except Exception:
                obj_tbl = None

            resolved_name = None
            ra_deg = np.nan
            dec_deg = np.nan

            if obj_tbl is not None and len(obj_tbl) > 0:
                row = obj_tbl[0]
                resolved_name = row["MAIN_ID"]
                if isinstance(resolved_name, bytes):
                    resolved_name = resolved_name.decode()
                if "RA_d" in obj_tbl.colnames:
                    ra_deg = float(row["RA_d"])
                if "DEC_d" in obj_tbl.colnames:
                    dec_deg = float(row["DEC_d"])

            gaia_id = extract_gaia_dr3_id_from_ids(ids_blob)

            return {
                "input_name": name,
                "resolved": True,
                "resolved_name": resolved_name if resolved_name is not None else cand,
                "matched_query_name": cand,
                "gaia_source_id": gaia_id if gaia_id is not None else pd.NA,
                "simbad_ra_deg": ra_deg,
                "simbad_dec_deg": dec_deg,
                "ids_blob": ids_blob,
                "resolve_note": "ok" if gaia_id is not None else "no_gaia_dr3_id",
            }

    return {
        "input_name": name,
        "resolved": False,
        "resolved_name": None,
        "matched_query_name": None,
        "gaia_source_id": pd.NA,
        "simbad_ra_deg": pd.NA,
        "simbad_dec_deg": pd.NA,
        "ids_blob": None,
        "resolve_note": "not_found_or_ambiguous",
    }


def resolve_name_list(names: list[str], debug: bool = False) -> pd.DataFrame:
    simbad = make_simbad()
    rows = []
    for name in names:
        rows.append(resolve_name(simbad, name, debug=debug))
    df = pd.DataFrame(rows)
    df["gaia_source_id"] = safe_source_id(df["gaia_source_id"])
    return df


# ----------------------------------------------------------------------
# Benchmark against catalog
# ----------------------------------------------------------------------

def benchmark_against_catalog(resolved: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    cat = catalog.copy()
    cat["source_id"] = safe_source_id(cat["source_id"])

    keep_cols = [
        "source_id",
        "catalog_rank",
        "final_score",
        "slope_absdiff",
        "tier",
        "quality_class",
        "passes_slope_le_0p01",
    ]
    keep_cols = [c for c in keep_cols if c in cat.columns]

    merged = resolved.merge(
        cat[keep_cols],
        how="left",
        left_on="gaia_source_id",
        right_on="source_id",
        suffixes=("", "_cat")
    )

    merged["in_final_catalog"] = merged["catalog_rank"].notna()

    ncat = len(cat)
    if "catalog_rank" in merged.columns:
        merged["catalog_percentile"] = np.where(
            merged["catalog_rank"].notna(),
            100.0 * merged["catalog_rank"] / ncat,
            np.nan
        )

    return merged


# ----------------------------------------------------------------------
# XP / plotting support
# ----------------------------------------------------------------------

def load_solar_reference(
    path: str, wave_grid: np.ndarray, *, normalize_ref_nm: float = 550.0
) -> np.ndarray:
    df = normalize_column_names(pd.read_csv(path))
    if "wavelength_nm" not in df.columns or "flux" not in df.columns:
        raise ValueError("Solar CSV must have columns wavelength_nm and flux")

    wave = df["wavelength_nm"].to_numpy(dtype=float)
    flux = df["flux"].to_numpy(dtype=float)

    m = np.isfinite(wave) & np.isfinite(flux)
    wave = wave[m]
    flux = flux[m]

    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]

    flux_grid = np.interp(wave_grid, wave, flux, left=np.nan, right=np.nan)
    return normalize_at_reference_wavelength(wave_grid, flux_grid, normalize_ref_nm)


def extract_flux_matrix(df: pd.DataFrame, sampling: np.ndarray) -> np.ndarray:
    df = normalize_column_names(df)

    if "flux" in df.columns:
        first_valid = None
        for val in df["flux"]:
            if val is not None:
                first_valid = val
                break
        if first_valid is not None and isinstance(first_valid, (list, tuple, np.ndarray)):
            flux_matrix = np.vstack([
                np.asarray(v, dtype=float) if v is not None else np.full(len(sampling), np.nan)
                for v in df["flux"]
            ])
            if flux_matrix.shape[1] == len(sampling):
                return flux_matrix

            # Cached XP batches may have been calibrated on a different grid
            # (e.g., 330-1050 nm). Remap onto requested sampling.
            src_n = flux_matrix.shape[1]
            src_grid = match_cached_xp_wave_grid(src_n, WSTEP)

            remapped = np.vstack([
                np.interp(sampling, src_grid, row, left=np.nan, right=np.nan)
                for row in flux_matrix
            ])
            return remapped

    possible = []
    for col in df.columns:
        s = str(col)
        if s == "source_id":
            continue
        try:
            float(s)
            possible.append(col)
        except Exception:
            pass

    if len(possible) == len(sampling):
        return df[possible].to_numpy(dtype=float)

    skip_cols = {
        "source_id", "solution_id", "designation",
        "flux", "flux_error", "fluxerr",
        "wavelength", "wavelength_nm",
    }

    numeric_cols = []
    for col in df.columns:
        if col in skip_cols:
            continue
        arr = pd.to_numeric(df[col], errors="coerce")
        if np.isfinite(arr).any():
            numeric_cols.append(col)

    if len(numeric_cols) >= len(sampling):
        return df[numeric_cols[:len(sampling)]].to_numpy(dtype=float)

    raise RuntimeError(f"Could not parse GaiaXPy flux columns. Columns: {list(df.columns)}")


def prepare_star_flux_like_gaia_stage2_aip(sampling: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """Match gaia_stage2_aip.score_calibrated_spectra: σ=1 Gauss smoothing then normalize-at-550."""
    arr = np.asarray(flux, dtype=float)
    arr = stage2.smooth_flux(arr, sigma_pix=1.0)
    return stage2.normalize_at_550nm(sampling, arr, window_nm=5.0)


def load_cached_xp_fluxes(
    xpdir: str,
    sampling: np.ndarray,
    source_ids: set[int],
    *,
    normalize_ref_nm: float = 550.0,
    preprocess_like_gaia_stage2_aip: bool = False,
) -> dict[int, np.ndarray]:
    xp_path = Path(xpdir)
    files = sorted(xp_path.glob("xp_batch_*.parquet"))
    flux_by_id: dict[int, np.ndarray] = {}
    saw_xpcorrect_column = False

    for fn in files:
        df = normalize_column_names(pd.read_parquet(fn))
        if "source_id" not in df.columns:
            continue
        if "xp_correction_applied" in df.columns:
            saw_xpcorrect_column = True

        ids = df["source_id"].astype(np.int64).to_numpy()
        need_mask = np.array([int(x) in source_ids for x in ids], dtype=bool)
        if not need_mask.any():
            continue

        sub = df.loc[need_mask].copy()
        flux_matrix = extract_flux_matrix(sub, sampling)
        sub_ids = sub["source_id"].astype(np.int64).to_numpy()

        for sid, flux in zip(sub_ids, flux_matrix):
            arr = np.asarray(flux, dtype=float)
            if preprocess_like_gaia_stage2_aip:
                flux_by_id[int(sid)] = prepare_star_flux_like_gaia_stage2_aip(sampling, arr)
            else:
                flux_by_id[int(sid)] = normalize_at_reference_wavelength(
                    sampling, arr, normalize_ref_nm
                )

    print(
        f"[XP cache check] xp_correction_applied column present: {saw_xpcorrect_column} "
        f"(loaded {len(flux_by_id)} source fluxes from {len(files)} batch files)"
    )
    return flux_by_id


def fetch_missing_literature_fluxes(
    source_ids: list[int],
    sampling: np.ndarray,
    *,
    normalize_ref_nm: float = 550.0,
    preprocess_like_gaia_stage2_aip: bool = False,
) -> dict[int, np.ndarray]:
    if calibrate is None:
        raise ImportError("gaiaxpy is required for --fetch-missing-lit")

    if not source_ids:
        return {}

    result = calibrate(source_ids, sampling=sampling, save_file=False)
    if isinstance(result, tuple):
        calibrated_df, returned_sampling = result
    else:
        calibrated_df = result
        returned_sampling = sampling

    calibrated_df = normalize_column_names(calibrated_df)
    flux_matrix = extract_flux_matrix(calibrated_df, sampling)
    ids = calibrated_df["source_id"].astype(np.int64).to_numpy()

    flux_by_id = {}
    for sid, flux in zip(ids, flux_matrix):
        arr = np.asarray(flux, dtype=float)
        if preprocess_like_gaia_stage2_aip:
            flux_by_id[int(sid)] = prepare_star_flux_like_gaia_stage2_aip(sampling, arr)
        else:
            flux_by_id[int(sid)] = normalize_at_reference_wavelength(
                sampling, arr, normalize_ref_nm
            )
    return flux_by_id


def mask_ranges_outside_wavelength_interval(
    wave_grid: np.ndarray, keep_lo: float, keep_hi: float
) -> list[tuple[float, float]]:
    """
    Closed intervals on the actual grid that lie strictly outside [keep_lo, keep_hi],
    for use with stage2.build_keep_mask (zero weighted metrics there).
    """
    w = np.asarray(wave_grid, dtype=float)
    wmin, wmax = float(np.nanmin(w)), float(np.nanmax(w))
    out: list[tuple[float, float]] = []
    left = w[w < keep_lo]
    if left.size:
        out.append((wmin, float(np.nanmax(left))))
    right = w[w > keep_hi]
    if right.size:
        out.append((float(np.nanmin(right)), wmax))
    return out


def _slope_twoband_bands_inside_window(keep_lo: float, keep_hi: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Median-flux blue/red bands for finite-difference slope, both inside [keep_lo, keep_hi]."""
    span = keep_hi - keep_lo
    if span < 80.0:
        raise ValueError(
            f"wavelength-range span {span:.1f} nm is too small; widen [lo, hi] for slope twobands."
        )
    blue_lo = keep_lo + 6.0
    blue_hi = keep_lo + 24.0
    red_hi = keep_hi - 4.0
    red_lo = keep_hi - 24.0
    if blue_hi >= red_lo:
        mid = 0.5 * (keep_lo + keep_hi)
        blue_hi = mid - 8.0
        red_lo = mid + 8.0
    if not (
        blue_lo < blue_hi < red_lo < red_hi
        and blue_lo >= keep_lo - 1e-9
        and red_hi <= keep_hi + 1e-9
    ):
        raise ValueError(
            "Could not place slope twobands inside wavelength-range; choose a wider interval."
        )
    return (blue_lo, blue_hi), (red_lo, red_hi)


def feature_ranges_inside_window(
    feature_ranges: Sequence[Tuple[float, float]],
    keep_lo: float,
    keep_hi: float,
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for flo, fhi in feature_ranges:
        lo = max(float(flo), keep_lo)
        hi = min(float(fhi), keep_hi)
        if hi > lo:
            out.append((lo, hi))
    return out


def spectrum_window_covers_grid(keep_lo: float, keep_hi: float) -> bool:
    """If True, use pipeline slope_between(450→900 nm) and full feature bands (only emission masks vary)."""
    twmin = float(stage2.DEFAULT_WAVE_MIN)
    twmax = float(stage2.DEFAULT_WAVE_MAX)
    return keep_lo <= twmin + 1e-6 and keep_hi >= twmax - 1e-6


def recompute_spectral_metrics_wavelength_interval(
    benchmark: pd.DataFrame,
    wave_grid: np.ndarray,
    solar_flux: np.ndarray,
    flux_by_id: dict[int, np.ndarray],
    keep_lo: float,
    keep_hi: float,
    slope_equivalent_fraction: float,
) -> pd.DataFrame:
    """Re-merge catalog benchmark with XP-based metrics restricted to [keep_lo, keep_hi] nm.

    Expects solar_flux and cached star flux prepared like gaia_stage2_aip.score_calibrated_spectra
    (smooth_flux σ=1, normalize-at-550 on the same wave_grid).
    """
    if keep_lo >= keep_hi:
        raise ValueError("--wavelength-range requires lo < hi (nanometers).")

    out = benchmark.copy()
    extra_mask = mask_ranges_outside_wavelength_interval(wave_grid, keep_lo, keep_hi)
    mask_ranges = list(stage2.DEFAULT_MASK_RANGES_NM) + extra_mask
    full_grid = spectrum_window_covers_grid(keep_lo, keep_hi)

    # Same feature downweight tuples as gaia_stage2 score_calibrated_spectra. Pixels outside
    # [--wavelength-range--] already have weight 0 from mask_ranges so extra feature rows
    # do not resurrect weight there (build_keep_mask only touches weights > 0).
    feature_ranges = list(stage2.DEFAULT_FEATURE_RANGES_NM)

    if full_grid:
        slope_sun = stage2.slope_between(
            wave_grid,
            solar_flux,
            stage2.SLOPE_FINITE_DIFF_LAM1_NM,
            stage2.SLOPE_FINITE_DIFF_LAM2_NM,
            half_window=stage2.SLOPE_FINITE_DIFF_HALF_WINDOW_NM,
        )

        def slope_star_fn(flux_arr: np.ndarray) -> float:
            return stage2.slope_between(
                wave_grid,
                flux_arr,
                stage2.SLOPE_FINITE_DIFF_LAM1_NM,
                stage2.SLOPE_FINITE_DIFF_LAM2_NM,
                half_window=stage2.SLOPE_FINITE_DIFF_HALF_WINDOW_NM,
            )
    else:
        (blue_b, red_b) = _slope_twoband_bands_inside_window(keep_lo, keep_hi)
        slope_sun = stage2.slope_twoband(wave_grid, solar_flux, *blue_b, *red_b)

        def slope_star_fn(flux_arr: np.ndarray) -> float:
            return stage2.slope_twoband(wave_grid, flux_arr, *blue_b, *red_b)

    weights = stage2.build_keep_mask(
        wave_grid, mask_ranges, feature_ranges, feature_weight=stage2.DEFAULT_SCORING_FEATURE_WEIGHT
    )

    feature_ranges_for_penalty = (
        feature_ranges
        if full_grid
        else feature_ranges_inside_window(stage2.DEFAULT_FEATURE_RANGES_NM, keep_lo, keep_hi)
    )

    wrms: list[float] = []
    sam: list[float] = []
    corr: list[float] = []
    slope_absdiff: list[float] = []
    feature_penalty: list[float] = []

    for _, row in out.iterrows():
        sid = row.get("gaia_source_id")
        if pd.isna(sid):
            wrms.append(np.nan)
            sam.append(np.nan)
            corr.append(np.nan)
            slope_absdiff.append(np.nan)
            feature_penalty.append(np.nan)
            continue

        sid_int = int(sid)
        flux = flux_by_id.get(sid_int)
        if flux is None:
            wrms.append(np.nan)
            sam.append(np.nan)
            corr.append(np.nan)
            slope_absdiff.append(np.nan)
            feature_penalty.append(np.nan)
            continue

        flux = np.asarray(flux, dtype=float)
        finite_fraction = float(np.mean(np.isfinite(flux)))
        if finite_fraction < 0.7:
            wrms.append(np.nan)
            sam.append(np.nan)
            corr.append(np.nan)
            slope_absdiff.append(np.nan)
            feature_penalty.append(np.nan)
            continue

        wrms.append(stage2.weighted_rms_residual(flux, solar_flux, weights))
        sam.append(stage2.spectral_angle(flux, solar_flux, weights))
        corr.append(stage2.weighted_corrcoef(flux, solar_flux, weights))
        slope_star = slope_star_fn(flux)
        slope_absdiff.append(
            abs(slope_star - slope_sun) if np.isfinite(slope_star) and np.isfinite(slope_sun) else np.nan
        )
        feature_penalty.append(
            stage2.feature_penalty(wave_grid, flux, solar_flux, feature_ranges_for_penalty)
        )

    out["wrms"] = wrms
    out["sam"] = sam
    out["corr"] = corr
    out["slope_absdiff"] = slope_absdiff
    out["feature_penalty"] = feature_penalty

    annotate_slope_equivalent_gate(out, slope_equivalent_fraction)
    return out


def make_plots(
    catalog: pd.DataFrame,
    benchmark: pd.DataFrame,
    solar_csv: str,
    xpdir: str,
    outdir: Path,
    fetch_missing_lit: bool = False,
) -> None:
    plot_norm_nm = 550.0
    plot_xlim = (400.0, 1000.0)
    plot_ylabel = "Normalized flux to 550 nm"

    def set_dynamic_ylim(ax, y_arrays: list[np.ndarray], pad_frac: float = 0.05):
        vals = []
        for y in y_arrays:
            arr = np.asarray(y, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                vals.append(arr)
        if not vals:
            return
        allv = np.concatenate(vals)
        ymin = float(np.nanmin(allv))
        ymax = float(np.nanmax(allv))
        if not np.isfinite(ymin) or not np.isfinite(ymax):
            return
        if ymax <= ymin:
            d = max(abs(ymin), 1.0) * 0.05
            ax.set_ylim(ymin - d, ymax + d)
            return
        pad = (ymax - ymin) * pad_frac
        ax.set_ylim(ymin - pad, ymax + pad)

    wave_grid = build_wave_grid()
    solar_flux = load_solar_reference(solar_csv, wave_grid, normalize_ref_nm=plot_norm_nm)

    lit_ids = set(
        pd.to_numeric(benchmark["gaia_source_id"], errors="coerce")
        .dropna().astype("int64").tolist()
    )

    flux_by_id = load_cached_xp_fluxes(
        xpdir, wave_grid, lit_ids, normalize_ref_nm=plot_norm_nm
    )

    if fetch_missing_lit:
        missing_lit = sorted([sid for sid in lit_ids if sid not in flux_by_id])
        if missing_lit:
            fetched = fetch_missing_literature_fluxes(
                missing_lit, wave_grid, normalize_ref_nm=plot_norm_nm
            )
            flux_by_id.update(fetched)

    mask_x = np.isfinite(wave_grid) & (wave_grid >= plot_xlim[0]) & (wave_grid <= plot_xlim[1])

    fig_h = 10.0 * (8.5 / 11.0)
    fig, (ax_passed, ax_all) = plt.subplots(2, 1, figsize=(8.5, fig_h), sharex=True, constrained_layout=True)

    y_passed = [solar_flux]
    ax_passed.plot(
        wave_grid[mask_x], solar_flux[mask_x], linewidth=2, color="black", linestyle="-"
    )
    passed = benchmark[benchmark["passes_slope_le_0p01"].fillna(False)].copy()
    for _, row in passed.iterrows():
        sid = row["gaia_source_id"]
        if pd.isnull(sid):
            continue
        sid = int(sid)
        if sid not in flux_by_id:
            continue
        y = flux_by_id[sid]
        y_passed.append(y)
        ax_passed.plot(wave_grid[mask_x], y[mask_x], linestyle="-", alpha=0.30)
    ax_passed.set_ylabel(plot_ylabel)
    ax_passed.set_xlim(plot_xlim)
    set_dynamic_ylim(ax_passed, y_passed)

    y_all = [solar_flux]
    ax_all.plot(
        wave_grid[mask_x], solar_flux[mask_x], linewidth=2, color="black", linestyle="-"
    )
    for _, row in benchmark.iterrows():
        sid = row["gaia_source_id"]
        if pd.isnull(sid):
            continue
        sid = int(sid)
        if sid not in flux_by_id:
            continue
        y = flux_by_id[sid]
        y_all.append(y)
        ax_all.plot(wave_grid[mask_x], y[mask_x], linestyle="-", alpha=0.10)
    ax_all.set_xlabel("Wavelength (nm)")
    ax_all.set_ylabel(plot_ylabel)
    ax_all.set_xlim(plot_xlim)
    set_dynamic_ylim(ax_all, y_all)

    plot_path_stem = outdir / "plot_literature_vs_solar"
    fig.savefig(f"{plot_path_stem}.png", dpi=200)
    fig.savefig(f"{plot_path_stem}.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def positive_slope_equivalent_percent(x: str) -> float:
    v = float(x)
    if v <= 0.0 or v > 100.0:
        raise argparse.ArgumentTypeError("--slope-equivalent-percent must be in (0, 100]")
    return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, help="Final ranked Gaia solar-analog catalog CSV")
    parser.add_argument("--names", required=True, help="Plain-text name list, one star per line")
    parser.add_argument("--outdir", default="literature_name_benchmark", help="Output directory")

    parser.add_argument("--solar-csv", default=None, help="Solar reference CSV for plotting / wavelength-range recompute")
    parser.add_argument("--xpdir", default=None, help="Directory containing cached xp_batch_*.parquet files")
    parser.add_argument(
        "--wavelength-range",
        nargs=2,
        type=float,
        metavar=("NM_LO", "NM_HI"),
        default=None,
        dest="wavelength_range",
        help=(
            "Recompute WRMS, SAM, corr, slope_absdiff, feature_penalty inside [NM_LO, NM_HI] nm only "
            "(weight=0 outside; requires --solar-csv and --xpdir). Example: --wavelength-range 550 900. "
            "Stars and solar use gaia_stage2_aip preprocessing (smooth_flux σ=1 px + normalize-at-550, "
            "same solar CSV path as stage2)."
        ),
    )
    parser.add_argument("--plot", action="store_true", help="Make spectra plots")
    parser.add_argument("--fetch-missing-lit", action="store_true",
                        help="Fetch Gaia XP for literature stars not in cache")
    parser.add_argument("--debug-resolve", action="store_true", help="Print SIMBAD resolution diagnostics")
    parser.add_argument(
        "--slope-equivalent-percent",
        type=positive_slope_equivalent_percent,
        default=1.0,
        metavar="PC",
        help=(
            "Percent-equivalent for the slope-absdiff gate (same convention as classify_catalog): "
            "e.g. 1 = 1%%, 0.5 = 0.5%%. Applies to catalog slope_absdiff or to recomputed slopes with "
            "--wavelength-range. Writes column slope_equivalent_percent and updates passes_slope_le_0p01."
        ),
    )

    args = parser.parse_args()
    slope_fraction = args.slope_equivalent_percent / 100.0

    if args.wavelength_range is not None:
        if args.solar_csv is None or args.xpdir is None:
            parser.error("--wavelength-range requires both --solar-csv and --xpdir")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    catalog = normalize_column_names(pd.read_csv(args.catalog))
    if "quality_class" not in catalog.columns:
        catalog = classify_catalog(catalog)

    names = load_name_list(args.names)
    resolved = resolve_name_list(names, debug=args.debug_resolve)
    resolved.to_csv(outdir / "resolved_name_list.csv", index=False)

    benchmark = benchmark_against_catalog(resolved, catalog)

    if args.wavelength_range is not None:
        wl_lo, wl_hi = float(args.wavelength_range[0]), float(args.wavelength_range[1])
        wave_grid = build_wave_grid()
        solar_flux = stage2.load_and_prepare_solar_reference(Path(args.solar_csv), wave_grid)
        lit_ids = set(
            pd.to_numeric(benchmark["gaia_source_id"], errors="coerce").dropna().astype("int64").tolist()
        )
        flux_by_id = load_cached_xp_fluxes(
            args.xpdir, wave_grid, lit_ids, preprocess_like_gaia_stage2_aip=True
        )
        if args.fetch_missing_lit:
            missing_lit = sorted([sid for sid in lit_ids if sid not in flux_by_id])
            if missing_lit:
                fetched = fetch_missing_literature_fluxes(
                    missing_lit, wave_grid, preprocess_like_gaia_stage2_aip=True
                )
                flux_by_id.update(fetched)
        benchmark = recompute_spectral_metrics_wavelength_interval(
            benchmark,
            wave_grid,
            solar_flux,
            flux_by_id,
            wl_lo,
            wl_hi,
            slope_fraction,
        )
        print("\n=== Wavelength restriction ===")
        print(f"  Recomputed metrics use λ in [{wl_lo:g}, {wl_hi:g}] nm (outside range has zero weight).")
        print(
            "  Solar + star spectra: gaia_stage2_aip.load_and_prepare_solar_reference "
            "and smooth_flux(σ=1)+normalize-at-550 (same chain as catalog scoring)."
        )
    else:
        annotate_slope_equivalent_gate(benchmark, slope_fraction)

    benchmark.to_csv(outdir / "literature_name_benchmark_vs_catalog.csv", index=False)

    print("\n=== Thresholds ===")
    cut_cli = slope_threshold_cut_from_percent(slope_fraction)
    print(
        " ",
        summarize_thresholds(
            "passes_slope_le_0p01 (column uses this cutoff)", slope_fraction, cut_cli
        ),
    )

    print("\n=== Resolution summary ===")
    print(f"Total names               : {len(names)}")
    print(f"Resolved by SIMBAD        : {int(resolved['resolved'].sum())}")
    print(f"With Gaia DR3 source_id   : {int(resolved['gaia_source_id'].notna().sum())}")

    print("\n=== Benchmark summary ===")
    print(f"In final catalog          : {int(benchmark['in_final_catalog'].sum())}")
    if "passes_slope_le_0p01" in benchmark.columns:
        n_pass = int(benchmark["passes_slope_le_0p01"].fillna(False).sum())
        print(
            f"passes_slope_le_0p01      "
            f": {n_pass}"
        )

        print("\n=== Stars passing passes_slope_le_0p01 ===")
        passed_rows = benchmark[benchmark["passes_slope_le_0p01"].fillna(False)]
        if passed_rows.empty:
            print("  (none)")
        else:
            passed_rows = passed_rows.sort_values("input_name", na_position="last")
            for _, row in passed_rows.iterrows():
                inm = str(row.get("input_name", "") or "").strip()
                rn = row.get("resolved_name")
                if pd.notna(rn) and str(rn).strip():
                    print(f"  {inm}  ({str(rn).strip()})")
                else:
                    print(f"  {inm}")

    if args.plot:
        if args.solar_csv is None or args.xpdir is None:
            raise ValueError("--plot requires both --solar-csv and --xpdir")
        make_plots(
            catalog=catalog,
            benchmark=benchmark,
            solar_csv=args.solar_csv,
            xpdir=args.xpdir,
            outdir=outdir,
            fetch_missing_lit=args.fetch_missing_lit,
        )

    print("\nWrote:")
    print("  ", outdir / "resolved_name_list.csv")
    print("  ", outdir / "literature_name_benchmark_vs_catalog.csv")
    if args.plot:
        print("  ", outdir / "plot_literature_vs_solar.png")
        print("  ", outdir / "plot_literature_vs_solar.pdf")


if __name__ == "__main__":
    main()
