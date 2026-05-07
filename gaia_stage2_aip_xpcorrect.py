#!/usr/bin/env python3
"""
gaia_stage2_aip_xpcorrect.py

Stage-2 Gaia@AIP pipeline variant that:
1) retrieves/calibrates Gaia DR3 XP spectra
2) applies GaiaXPcorrection to each calibrated spectrum
3) scores corrected spectra against a solar reference on 400-1050 nm only
4) writes XP/score batches and assembles final ranked catalog
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import gaia_stage2_aip as base

# Update this to your local GaiaXPcorrection checkout root (dummy placeholder).
# It should be the directory that makes this import work:
# from GaiaDR3XPspectracorrectionV1 import Gaia_Correction_V1
LOCAL_GAIAXP_CORRECTION_PATH = "/home/qye/Library/GaiaXPcorrection"

if LOCAL_GAIAXP_CORRECTION_PATH and LOCAL_GAIAXP_CORRECTION_PATH not in sys.path:
    sys.path.insert(0, LOCAL_GAIAXP_CORRECTION_PATH)

try:
    from GaiaDR3XPspectracorrectionV1 import Gaia_Correction_V1
except Exception:
    Gaia_Correction_V1 = None


def extract_flux_error_matrix(calibrated_df: pd.DataFrame, sampling: np.ndarray) -> np.ndarray:
    df = base.normalize_column_names(calibrated_df)
    n = len(df)
    m = len(sampling)

    for col in ["flux_error", "fluxerr"]:
        if col not in df.columns:
            continue
        first_valid = None
        for val in df[col]:
            if val is not None:
                first_valid = val
                break
        if first_valid is not None and isinstance(first_valid, (list, tuple, np.ndarray)):
            mat = np.vstack([
                np.asarray(v, dtype=float) if v is not None else np.full(m, np.nan)
                for v in df[col]
            ])
            if mat.shape[1] == m:
                return mat
    return np.full((n, m), np.nan, dtype=float)


def apply_xp_correction(
    calibrated_df: pd.DataFrame,
    gmag_by_source: dict[int, float],
    sampling: np.ndarray,
) -> pd.DataFrame:
    if Gaia_Correction_V1 is None:
        raise ImportError(
            "GaiaXPcorrection is not installed. Install from "
            "https://github.com/HiromonGON/GaiaXPcorrection"
        )

    df = base.normalize_column_names(calibrated_df.copy())
    if "source_id" not in df.columns:
        raise ValueError("Calibrated XP table must contain source_id")

    flux_matrix = base.extract_flux_matrix(df, sampling)
    err_matrix = extract_flux_error_matrix(df, sampling)
    source_ids = pd.to_numeric(df["source_id"], errors="coerce").astype("Int64").to_numpy()

    corrected = []
    cautions = []
    applied = []

    for sid, flux, err in zip(source_ids, flux_matrix, err_matrix):
        sid_int = int(sid) if pd.notna(sid) else None
        gmag = gmag_by_source.get(sid_int, np.nan)
        flux = np.asarray(flux, dtype=float)
        err = np.asarray(err, dtype=float)

        if sid_int is None or not np.isfinite(gmag):
            corrected.append(flux.copy())
            cautions.append(np.nan)
            applied.append(False)
            continue

        try:
            try:
                flux_out, caution, _, _ = Gaia_Correction_V1.correction(
                    flux,
                    float(gmag),
                    err,
                    Truncation=False,
                    absolute_correction=True,
                )
            except TypeError:
                flux_out, caution, _, _ = Gaia_Correction_V1.correction(flux, float(gmag), err)
            out = np.asarray(flux_out, dtype=float)
            if out.shape != flux.shape:
                out = flux.copy()
                caution = np.nan
                was_applied = False
            else:
                was_applied = True
            corrected.append(out)
            cautions.append(caution)
            applied.append(was_applied)
        except Exception:
            corrected.append(flux.copy())
            cautions.append(np.nan)
            applied.append(False)

    df["flux"] = corrected
    df["xp_correction_caution"] = cautions
    df["xp_correction_applied"] = applied
    return df


def score_corrected_spectra_400_1050(
    calibrated_df: pd.DataFrame,
    sampling: np.ndarray,
    solar_flux: np.ndarray,
) -> pd.DataFrame:
    if calibrated_df.empty:
        return pd.DataFrame(columns=[
            "source_id", "xp_finite_fraction", "wrms", "sam", "corr",
            "slope_star", "slope_sun", "slope_absdiff", "feature_penalty"
        ])

    sampling = np.asarray(sampling, dtype=float)
    use = (sampling >= 400.0) & (sampling <= 1050.0)
    if use.sum() < 10:
        raise ValueError("Not enough sampling points in 400-1050 nm for scoring")

    sampling_use = sampling[use]
    solar_use = np.asarray(solar_flux, dtype=float)[use]
    flux_matrix = base.extract_flux_matrix(calibrated_df, sampling)[:, use]
    source_ids = calibrated_df["source_id"].astype(np.int64).to_numpy()

    mask_ranges = [(686.0, 696.0), (716.0, 734.0), (758.0, 771.0), (810.0, 840.0), (930.0, 970.0)]
    feature_ranges = [(426.0, 434.0), (484.0, 488.0), (515.0, 521.0), (654.0, 659.0)]
    weights = base.build_keep_mask(sampling_use, mask_ranges, feature_ranges, feature_weight=0.35)

    rows = []
    for sid, flux in zip(source_ids, flux_matrix):
        flux = np.asarray(flux, dtype=float)
        flux = base.smooth_flux(flux, sigma_pix=1.0)
        flux = base.normalize_at_550nm(sampling_use, flux, window_nm=5.0)

        finite_fraction = float(np.mean(np.isfinite(flux)))
        if finite_fraction < 0.7:
            rows.append({
                "source_id": sid,
                "xp_finite_fraction": finite_fraction,
                "wrms": np.nan,
                "sam": np.nan,
                "corr": np.nan,
                "slope_star": np.nan,
                "slope_sun": np.nan,
                "slope_absdiff": np.nan,
                "feature_penalty": np.nan,
            })
            continue

        wrms = base.weighted_rms_residual(flux, solar_use, weights)
        sam = base.spectral_angle(flux, solar_use, weights)
        corr = base.weighted_corrcoef(flux, solar_use, weights)
        slope_star = base.slope_between(sampling_use, flux, 450.0, 900.0, half_window=5.0)
        slope_sun = base.slope_between(sampling_use, solar_use, 450.0, 900.0, half_window=5.0)
        slope_absdiff = abs(slope_star - slope_sun) if np.isfinite(slope_star) and np.isfinite(slope_sun) else np.nan
        feat_pen = base.feature_penalty(sampling_use, flux, solar_use, feature_ranges)

        rows.append({
            "source_id": sid,
            "xp_finite_fraction": finite_fraction,
            "wrms": wrms,
            "sam": sam,
            "corr": corr,
            "slope_star": slope_star,
            "slope_sun": slope_sun,
            "slope_absdiff": slope_absdiff,
            "feature_penalty": feat_pen,
        })

    return pd.DataFrame(rows)


def process_stage2_batches_xpcorrect(
    candidates: pd.DataFrame,
    wave_grid: np.ndarray,
    solar_flux: np.ndarray,
    outdir: Path,
    batch_size: int,
    sleep_s: float,
    max_retries: int,
    aip_token: str,
    aip_tap_url: str,
    aip_sjs_url: str,
    aip_queue: str,
    retrieve_only: bool = False,
    score_only: bool = False,
) -> None:
    if not score_only and base.calibrate is None:
        raise ImportError("gaiaxpy is not installed. Please: pip install gaiaxpy")

    xp_dir = outdir / "xp_batches"
    raw_dir = xp_dir / "raw_aip"
    score_dir = outdir / "score_batches"
    xp_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    gmag_by_source = dict(
        zip(
            pd.to_numeric(candidates["source_id"], errors="coerce").astype("Int64").astype(np.int64),
            pd.to_numeric(candidates.get("phot_g_mean_mag"), errors="coerce"),
        )
    )

    failed_rows = []
    deferred_rows = []

    def try_calibrate_batch(batch_df: pd.DataFrame, label: str):
        last_err = None
        saw_transient = False

        for attempt in range(1, max_retries + 1):
            try:
                calibrated_df = base.fetch_one_batch_aip(
                    batch_df=batch_df,
                    wave_grid=wave_grid,
                    raw_dir=raw_dir,
                    batch_label=label,
                    token=aip_token,
                    tap_url=aip_tap_url,
                    sjs_url=aip_sjs_url,
                    queue=aip_queue,
                )
                return calibrated_df, None, saw_transient
            except Exception as e:
                last_err = e
                print(f"  {label} failed on attempt {attempt}/{max_retries}: {repr(e)}", file=sys.stderr)
                traceback.print_exc()
                if base.is_transient_xp_error(e):
                    saw_transient = True
                if base.is_ambiguous_no_xp_error(e) and len(batch_df) > 1:
                    break
                if attempt < max_retries:
                    wait_s = base.retry_wait_time(sleep_s, attempt)
                    print(f"  Retrying in {wait_s:.1f} s...", file=sys.stderr)
                    time.sleep(wait_s)
        return None, last_err, saw_transient

    def score_and_save(calibrated_df: pd.DataFrame, score_file: Path, batch_label: str):
        if score_file.exists():
            print(f"Skipping scoring for {batch_label}: already exists", file=sys.stderr)
            return
        corrected_df = apply_xp_correction(calibrated_df, gmag_by_source=gmag_by_source, sampling=wave_grid)
        score_df = score_corrected_spectra_400_1050(
            corrected_df,
            sampling=wave_grid,
            solar_flux=solar_flux,
        )
        base.safe_to_parquet(score_df, score_file, index=False)
        print(f"Scored {batch_label}", file=sys.stderr)

    def retrieve_recursive(batch_df: pd.DataFrame, label: str):
        source_ids = batch_df["source_id"].astype(np.int64).tolist()
        calibrated_df, err, saw_transient = try_calibrate_batch(batch_df, label)
        if calibrated_df is not None:
            return [calibrated_df], [], []

        if len(source_ids) > 1 and base.is_ambiguous_no_xp_error(err):
            mid = len(batch_df) // 2
            left = batch_df.iloc[:mid].copy()
            right = batch_df.iloc[mid:].copy()
            print(f"  Splitting {label} into {len(left)} + {len(right)}", file=sys.stderr)
            rec_l, fail_l, defer_l = retrieve_recursive(left, f"{label}a")
            rec_r, fail_r, defer_r = retrieve_recursive(right, f"{label}b")
            return rec_l + rec_r, fail_l + fail_r, defer_l + defer_r

        if len(source_ids) > 1 and (saw_transient or base.is_transient_xp_error(err)):
            mid = len(batch_df) // 2
            left = batch_df.iloc[:mid].copy()
            right = batch_df.iloc[mid:].copy()
            print(f"  Transient failure: splitting {label} into {len(left)} + {len(right)}", file=sys.stderr)
            rec_l, fail_l, defer_l = retrieve_recursive(left, f"{label}a")
            rec_r, fail_r, defer_r = retrieve_recursive(right, f"{label}b")
            return rec_l + rec_r, fail_l + fail_r, defer_l + defer_r

        sid = int(source_ids[0])
        if base.is_transient_xp_error(err):
            print(f"  Deferring source {sid} after transient server failures", file=sys.stderr)
            return [], [], [sid]
        if base.is_ambiguous_no_xp_error(err):
            print(f"  Marking source {sid} as failed: no continuous XP data returned", file=sys.stderr)
            return [], [sid], []
        print(f"  Marking source {sid} as failed: {err}", file=sys.stderr)
        return [], [sid], []

    total = len(candidates)
    for i in range(0, total, batch_size):
        batch_df = candidates.iloc[i:i + batch_size].copy()
        batch_num = i // batch_size + 1

        xp_file = xp_dir / f"xp_batch_{batch_num:05d}.parquet"
        score_file = score_dir / f"score_batch_{batch_num:05d}.parquet"
        fail_file = xp_dir / f"xp_batch_{batch_num:05d}_failed.csv"
        defer_file = xp_dir / f"xp_batch_{batch_num:05d}_deferred.csv"

        if score_only:
            if not xp_file.exists():
                print(f"Skipping batch {batch_num}: no cached XP file", file=sys.stderr)
                continue
            if score_file.exists():
                print(f"Skipping batch {batch_num}: score batch already exists", file=sys.stderr)
                continue
            calibrated_df = base.normalize_column_names(pd.read_parquet(xp_file))
            score_and_save(calibrated_df, score_file, f"cached batch {batch_num}")
            continue

        if xp_file.exists():
            print(f"Loading cached XP batch {batch_num}", file=sys.stderr)
            calibrated_df = base.normalize_column_names(pd.read_parquet(xp_file))
            if not retrieve_only:
                if score_file.exists():
                    print(f"Skipping scoring for batch {batch_num}: already exists", file=sys.stderr)
                else:
                    score_and_save(calibrated_df, score_file, f"batch {batch_num}")
            continue

        print(f"Fetching/calibrating Gaia@AIP XP batch {batch_num}: {len(batch_df)} sources", file=sys.stderr)
        recovered_dfs, batch_failed, batch_deferred = retrieve_recursive(batch_df, f"Batch_{batch_num:05d}")

        if batch_failed:
            batch_failed = sorted(set(int(x) for x in batch_failed))
            failed_rows.extend(batch_failed)
            pd.DataFrame({"source_id": batch_failed}).to_csv(fail_file, index=False)
        if batch_deferred:
            batch_deferred = sorted(set(int(x) for x in batch_deferred))
            deferred_rows.extend(batch_deferred)
            pd.DataFrame({"source_id": batch_deferred}).to_csv(defer_file, index=False)
        if not recovered_dfs:
            print(f"Batch {batch_num}: nothing recovered", file=sys.stderr)
            continue

        calibrated_df = pd.concat(recovered_dfs, ignore_index=True)
        calibrated_df = base.normalize_column_names(calibrated_df).drop_duplicates(subset=["source_id"]).reset_index(drop=True)
        calibrated_df = apply_xp_correction(calibrated_df, gmag_by_source=gmag_by_source, sampling=wave_grid)
        base.safe_to_parquet(calibrated_df, xp_file, index=False)

        if retrieve_only:
            continue
        if score_file.exists():
            print(f"Skipping scoring for batch {batch_num}: already exists", file=sys.stderr)
            continue
        score_and_save(calibrated_df, score_file, f"batch {batch_num}")

    if failed_rows:
        failed_rows = sorted(set(int(x) for x in failed_rows))
        pd.DataFrame({"source_id": failed_rows}).to_csv(xp_dir / "all_failed_source_ids.csv", index=False)
    if deferred_rows:
        deferred_rows = sorted(set(int(x) for x in deferred_rows))
        pd.DataFrame({"source_id": deferred_rows}).to_csv(xp_dir / "all_deferred_source_ids.csv", index=False)
        print(f"Wrote {len(deferred_rows)} deferred source IDs to {xp_dir / 'all_deferred_source_ids.csv'}", file=sys.stderr)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-2 Gaia solar-analog pipeline using Gaia@AIP + GaiaXPcorrection")
    p.add_argument("--stage1-csv", type=str, required=True, help="Stage-1 candidate CSV")
    p.add_argument("--solar-csv", type=str, required=True, help="Solar reference CSV with wavelength_nm, flux")
    p.add_argument("--outdir", type=str, default="gaia_solar_analogs_out_aip_xpcorrect")
    p.add_argument("--retrieve-only", action="store_true", help="Retrieve/cache XP batches only; do not score")
    p.add_argument("--score-only", action="store_true", help="Score existing cached XP batches only; do not retrieve")
    p.add_argument("--wave-min", type=float, default=base.DEFAULT_WAVE_MIN)
    p.add_argument("--wave-max", type=float, default=base.DEFAULT_WAVE_MAX)
    p.add_argument("--wave-step", type=float, default=base.DEFAULT_WAVE_STEP)
    p.add_argument("--max-sources", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--sleep-s", type=float, default=10.0)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--aip-token", type=str, required=True, help="Gaia@AIP API token")
    p.add_argument("--aip-tap-url", type=str, default="https://gaia.aip.de/tap")
    p.add_argument("--aip-sjs-url", type=str, default="https://gaia.aip.de/uws/simple-join-service")
    p.add_argument("--aip-queue", type=str, default="30s", choices=["30s", "5m", "2h"])
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.retrieve_only and args.score_only:
        raise ValueError("Use at most one of --retrieve-only and --score-only")
    if base.calibrate is None and not args.score_only:
        raise ImportError("gaiaxpy is not installed. Please: pip install gaiaxpy")
    if Gaia_Correction_V1 is None:
        raise ImportError(
            "GaiaXPcorrection is not installed. Install from "
            "https://github.com/HiromonGON/GaiaXPcorrection"
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    wave_grid = base.build_wave_grid(args.wave_min, args.wave_max, args.wave_step)
    if np.nanmin(wave_grid) < 330.0 or np.nanmax(wave_grid) > 1050.0:
        raise ValueError(
            f"GaiaXPy sampling must stay within 330-1050 nm. Got {np.nanmin(wave_grid)}-{np.nanmax(wave_grid)} nm."
        )

    candidates = base.normalize_column_names(pd.read_csv(args.stage1_csv))
    if "source_id" not in candidates.columns:
        raise ValueError("Stage-1 CSV must contain a source_id column")
    candidates["source_id"] = pd.to_numeric(candidates["source_id"], errors="coerce").astype("Int64")
    candidates = candidates[candidates["source_id"].notna()].copy()
    candidates["source_id"] = candidates["source_id"].astype(np.int64)
    if args.max_sources is not None and args.max_sources > 0:
        candidates = candidates.iloc[:args.max_sources].copy()
    print(f"Loaded stage-1 table: {args.stage1_csv} ({len(candidates)} rows)", file=sys.stderr)

    solar_flux = base.load_and_prepare_solar_reference(Path(args.solar_csv), wave_grid)
    process_stage2_batches_xpcorrect(
        candidates=candidates,
        wave_grid=wave_grid,
        solar_flux=solar_flux,
        outdir=outdir,
        batch_size=args.batch_size,
        sleep_s=args.sleep_s,
        max_retries=args.max_retries,
        aip_token=args.aip_token,
        aip_tap_url=args.aip_tap_url,
        aip_sjs_url=args.aip_sjs_url,
        aip_queue=args.aip_queue,
        retrieve_only=args.retrieve_only,
        score_only=args.score_only,
    )

    if not args.retrieve_only:
        base.assemble_final_catalog(candidates, outdir)


if __name__ == "__main__":
    main()
