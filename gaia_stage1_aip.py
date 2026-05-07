#!/usr/bin/env python3
"""
gaia_stage1_aip.py

Stage-1-only Gaia DR3 candidate builder using Gaia@AIP TAP.
Equivalent intent to: python gaia.py --stage1-only
but without astroquery/Gaia ESA Archive dependency.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
from io import StringIO
from pathlib import Path
import re
import time
from typing import Optional, Sequence

import pandas as pd
import requests
from astropy.table import Table


DEFAULT_TEFF_MIN = 5500.0
DEFAULT_TEFF_MAX = 6300.0
DEFAULT_LOGG_MIN = 3.25
DEFAULT_LOGG_MAX = 4.46
DEFAULT_MH_MIN = -1.20
DEFAULT_MH_MAX = 0.30
DEFAULT_RUWE_MAX = 1.25
DEFAULT_PLX_OVER_ERR_MIN = 100.0
DEFAULT_GMAG_MAX = 10.0
DEFAULT_ASTROMETRIC_EXCESS_MAX = 5.0
DEFAULT_PHOT_BP_RP_EXCESS_MAX = 1.3
DEFAULT_PHOT_BP_RP_EXCESS_MIN = 1.0
DEFAULT_EBPMINRP_MAX = 0.06


@dataclass
class QueryConfig:
    teff_min: float = DEFAULT_TEFF_MIN
    teff_max: float = DEFAULT_TEFF_MAX
    logg_min: float = DEFAULT_LOGG_MIN
    logg_max: float = DEFAULT_LOGG_MAX
    mh_min: float = DEFAULT_MH_MIN
    mh_max: float = DEFAULT_MH_MAX
    ruwe_max: float = DEFAULT_RUWE_MAX
    plx_over_err_min: float = DEFAULT_PLX_OVER_ERR_MIN
    gmag_max: float = DEFAULT_GMAG_MAX
    ebpminrp_max: float = DEFAULT_EBPMINRP_MAX
    astrometric_excess_max: float = DEFAULT_ASTROMETRIC_EXCESS_MAX
    phot_bp_rp_excess_min: float = DEFAULT_PHOT_BP_RP_EXCESS_MIN
    phot_bp_rp_excess_max: float = DEFAULT_PHOT_BP_RP_EXCESS_MAX
    limit: int = -1


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def ensure_source_id_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "source_id" in out.columns:
        return out
    if "datalinkid" in out.columns:
        return out.rename(columns={"datalinkid": "source_id"})
    if "sourceid" in out.columns:
        return out.rename(columns={"sourceid": "source_id"})
    for col in out.columns:
        c = str(col).lower()
        if "source_id" in c or c.endswith("sourceid"):
            return out.rename(columns={col: "source_id"})
    return out


def make_stage1_adql(cfg: QueryConfig) -> str:
    top_clause = "" if cfg.limit is None or cfg.limit < 0 else f"TOP {int(cfg.limit)}"
    return f"""
SELECT {top_clause}
    gs.source_id,
    gs.ra,
    gs.dec,
    gs.phot_g_mean_mag,
    gs.phot_bp_mean_mag,
    gs.phot_rp_mean_mag,
    gs.bp_rp,
    gs.parallax,
    gs.parallax_error,
    gs.parallax_over_error,
    gs.pmra,
    gs.pmdec,
    gs.ruwe,
    gs.has_xp_continuous,
    gs.phot_bp_rp_excess_factor,
    gs.astrometric_excess_noise,
    gs.non_single_star,
    gs.in_qso_candidates,
    gs.in_galaxy_candidates,
    ap.teff_gspphot,
    ap.logg_gspphot,
    ap.mh_gspphot,
    ap.ag_gspphot,
    ap.ebpminrp_gspphot,
    ap.libname_gspphot
FROM gaiadr3.gaia_source AS gs
JOIN gaiadr3.astrophysical_parameters AS ap
    ON gs.source_id = ap.source_id
WHERE
    gs.has_xp_continuous = true
    AND gs.phot_g_mean_mag < {cfg.gmag_max}
    AND ap.teff_gspphot BETWEEN {cfg.teff_min} AND {cfg.teff_max}
    AND ap.logg_gspphot BETWEEN {cfg.logg_min} AND {cfg.logg_max}
    AND ap.mh_gspphot BETWEEN {cfg.mh_min} AND {cfg.mh_max}
    AND gs.ruwe < {cfg.ruwe_max}
    AND gs.parallax_over_error >= {cfg.plx_over_err_min}
    AND gs.astrometric_excess_noise < {cfg.astrometric_excess_max}
    AND gs.phot_bp_rp_excess_factor BETWEEN {cfg.phot_bp_rp_excess_min} AND {cfg.phot_bp_rp_excess_max}
    AND ap.ebpminrp_gspphot < {cfg.ebpminrp_max}
    AND gs.in_qso_candidates = false
    AND gs.in_galaxy_candidates = false
""".strip()


def run_aip_tap_query(
    query: str,
    tap_sync_url: str,
    aip_token: str | None = None,
    timeout_s: int = 600,
    tap_async_url: str | None = None,
    aip_queue: str = "2h",
    poll_interval_s: float = 5.0,
    async_timeout_s: int = 7200,
) -> pd.DataFrame:
    def _parse_table_text(text: str) -> pd.DataFrame:
        text = text.strip()
        if not text:
            return pd.DataFrame()
        if text.startswith("<?xml"):
            m = re.search(r'QUERY_STATUS"\s+value="([^"]+)"', text, flags=re.IGNORECASE)
            if m and m.group(1).strip().upper() == "ERROR":
                raise RuntimeError(f"AIP TAP query returned an error VOTable:\n{text[:2000]}")
            table = Table.read(BytesIO(text.encode("utf-8")), format="votable")
            return table.to_pandas()
        return pd.read_csv(StringIO(text))

    def _run_async_query() -> pd.DataFrame:
        if not tap_async_url:
            raise RuntimeError(
                "AIP TAP sync query timed out, but no async endpoint is configured. "
                "Set --aip-tap-async-url."
            )
        submit_payload = {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "csv",
            "QUERY": query,
            "QUEUE": aip_queue,
        }
        submit = requests.post(tap_async_url, data=submit_payload, headers=headers, timeout=timeout_s, allow_redirects=False)
        if submit.status_code not in (200, 303):
            raise RuntimeError(
                f"AIP TAP async submit failed with HTTP {submit.status_code}.\n"
                f"Response body:\n{(submit.text or '').strip()[:4000]}"
            )

        job_url = submit.headers.get("Location") or submit.headers.get("location")
        if not job_url:
            m = re.search(r"<uws:jobId>([^<]+)</uws:jobId>", submit.text or "", flags=re.IGNORECASE)
            if m:
                job_url = tap_async_url.rstrip("/") + "/" + m.group(1).strip()
        if not job_url:
            raise RuntimeError(
                "Could not determine async TAP job URL from submission response."
            )

        run_resp = requests.post(
            f"{job_url.rstrip('/')}/phase",
            data={"PHASE": "RUN"},
            headers=headers,
            timeout=timeout_s,
        )
        if run_resp.status_code not in (200, 303):
            raise RuntimeError(
                f"AIP TAP async run request failed with HTTP {run_resp.status_code}.\n"
                f"Response body:\n{(run_resp.text or '').strip()[:4000]}"
            )

        t0 = time.time()
        phase_url = f"{job_url.rstrip('/')}/phase"
        while True:
            phase_resp = requests.get(phase_url, headers=headers, timeout=timeout_s)
            phase_resp.raise_for_status()
            phase = (phase_resp.text or "").strip().upper()
            if phase == "COMPLETED":
                break
            if phase in {"ERROR", "ABORTED"}:
                err_resp = requests.get(f"{job_url.rstrip('/')}/error", headers=headers, timeout=timeout_s)
                raise RuntimeError(
                    f"AIP TAP async job ended in phase {phase}.\n"
                    f"Error body:\n{(err_resp.text or '').strip()[:4000]}"
                )
            if (time.time() - t0) > async_timeout_s:
                raise TimeoutError(
                    f"AIP TAP async job exceeded timeout ({async_timeout_s}s). Job URL: {job_url}"
                )
            time.sleep(poll_interval_s)

        result_resp = requests.get(f"{job_url.rstrip('/')}/results/result", headers=headers, timeout=timeout_s)
        result_resp.raise_for_status()
        return _parse_table_text(result_resp.text)

    payload = {
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "csv",
        "QUERY": query,
    }
    headers = {}
    if aip_token:
        headers["Authorization"] = f"Token {aip_token}"

    response = requests.post(tap_sync_url, data=payload, headers=headers, timeout=timeout_s)
    if not response.ok:
        snippet = (response.text or "").strip()
        if response.status_code == 400 and "statement timeout" in snippet.lower():
            return _run_async_query()
        raise RuntimeError(
            f"AIP TAP HTTP {response.status_code} error for {tap_sync_url}.\n"
            f"Response body:\n{snippet[:4000]}"
        )

    return _parse_table_text(response.text)


def query_stage1_candidates(
    cfg: QueryConfig,
    tap_sync_url: str,
    tap_async_url: str,
    aip_token: str | None,
    timeout_s: int,
    aip_queue: str,
    poll_interval_s: float,
    async_timeout_s: int,
    dump_to: Optional[Path] = None,
) -> pd.DataFrame:
    adql = make_stage1_adql(cfg)
    df = run_aip_tap_query(
        query=adql,
        tap_sync_url=tap_sync_url,
        aip_token=aip_token,
        timeout_s=timeout_s,
        tap_async_url=tap_async_url,
        aip_queue=aip_queue,
        poll_interval_s=poll_interval_s,
        async_timeout_s=async_timeout_s,
    )
    df = normalize_column_names(df)
    df = ensure_source_id_column(df)

    bool_cols = ["has_xp_continuous", "non_single_star", "in_qso_candidates", "in_galaxy_candidates"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "t", "1"])

    if "non_single_star" in df.columns:
        df = df[(~df["non_single_star"]) | (df["non_single_star"].isna())].copy()

    df["stage1_distance_from_solar"] = (
        ((df["teff_gspphot"] - 5772.0) / 120.0) ** 2
        + ((df["logg_gspphot"] - 4.44) / 0.20) ** 2
        + ((df["mh_gspphot"] - 0.00) / 0.15) ** 2
    )
    df = df.sort_values(["stage1_distance_from_solar", "phot_g_mean_mag"], ascending=[True, True]).reset_index(drop=True)
    if "source_id" in df.columns:
        ordered = ["source_id"] + [c for c in df.columns if c != "source_id"]
        df = df[ordered]

    if dump_to is not None:
        dump_to.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dump_to, index=False)

    return df


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build stage-1 Gaia solar-analog candidates with Gaia@AIP")
    p.add_argument("--outdir", type=str, default="gaia_solar_analogs_out")
    p.add_argument("--output-csv", type=str, default="gaia_stage1_candidates.csv")
    p.add_argument("--aip-tap-sync-url", type=str, default="https://gaia.aip.de/tap/sync")
    p.add_argument("--aip-tap-async-url", type=str, default="https://gaia.aip.de/tap/async")
    p.add_argument("--aip-token", type=str, default=None, help="Gaia@AIP API token")
    p.add_argument("--timeout-s", type=int, default=600, help="HTTP timeout for TAP sync query")
    p.add_argument("--aip-queue", type=str, default="2h", choices=["30s", "5m", "2h"])
    p.add_argument("--poll-interval-s", type=float, default=5.0)
    p.add_argument("--async-timeout-s", type=int, default=7200)

    p.add_argument("--gmag-max", type=float, default=DEFAULT_GMAG_MAX)
    p.add_argument("--teff-min", type=float, default=DEFAULT_TEFF_MIN)
    p.add_argument("--teff-max", type=float, default=DEFAULT_TEFF_MAX)
    p.add_argument("--logg-min", type=float, default=DEFAULT_LOGG_MIN)
    p.add_argument("--logg-max", type=float, default=DEFAULT_LOGG_MAX)
    p.add_argument("--mh-min", type=float, default=DEFAULT_MH_MIN)
    p.add_argument("--mh-max", type=float, default=DEFAULT_MH_MAX)
    p.add_argument("--ruwe-max", type=float, default=DEFAULT_RUWE_MAX)
    p.add_argument("--plx-over-err-min", type=float, default=DEFAULT_PLX_OVER_ERR_MIN)
    p.add_argument("--ebpminrp-max", type=float, default=DEFAULT_EBPMINRP_MAX)
    p.add_argument("--astrometric-excess-max", type=float, default=DEFAULT_ASTROMETRIC_EXCESS_MAX)
    p.add_argument("--phot-bp-rp-excess-min", type=float, default=DEFAULT_PHOT_BP_RP_EXCESS_MIN)
    p.add_argument("--phot-bp-rp-excess-max", type=float, default=DEFAULT_PHOT_BP_RP_EXCESS_MAX)
    p.add_argument("--limit", type=int, default=-1)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / args.output_csv

    cfg = QueryConfig(
        teff_min=args.teff_min,
        teff_max=args.teff_max,
        logg_min=args.logg_min,
        logg_max=args.logg_max,
        mh_min=args.mh_min,
        mh_max=args.mh_max,
        ruwe_max=args.ruwe_max,
        plx_over_err_min=args.plx_over_err_min,
        gmag_max=args.gmag_max,
        ebpminrp_max=args.ebpminrp_max,
        astrometric_excess_max=args.astrometric_excess_max,
        phot_bp_rp_excess_min=args.phot_bp_rp_excess_min,
        phot_bp_rp_excess_max=args.phot_bp_rp_excess_max,
        limit=args.limit,
    )

    df = query_stage1_candidates(
        cfg=cfg,
        tap_sync_url=args.aip_tap_sync_url,
        tap_async_url=args.aip_tap_async_url,
        aip_token=args.aip_token,
        timeout_s=int(args.timeout_s),
        aip_queue=args.aip_queue,
        poll_interval_s=float(args.poll_interval_s),
        async_timeout_s=int(args.async_timeout_s),
        dump_to=output_path,
    )
    print(f"Wrote stage-1 table: {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
