#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astroquery.simbad import Simbad

import gaia_stage2_aip as stage2

CONTINUUM_WINDOWS_NM = stage2.CONTINUUM_WINDOWS_NM


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def continuum_window_centers(windows):
    return np.array([(lo + hi) / 2.0 for lo, hi in windows], dtype=float)


def slope_thresholds_from_windows(windows):
    centers = continuum_window_centers(windows)
    delta_lambda = float(np.max(centers) - np.min(centers))
    cut_5pct = 0.05 / delta_lambda
    cut_1pct = 0.01 / delta_lambda
    return delta_lambda, cut_5pct, cut_1pct


def make_simbad():
    s = Simbad()
    s.add_votable_fields("ids", "sp")
    return s


def canonical_name(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("−", "-")
    s = re.sub(r"\s+", " ", s.strip().upper())
    return s


def strip_prefix(name: str) -> str:
    n = canonical_name(name)
    for p in ["NAME ", "* ", "GAIA DR3 ", "GAIA DR2 "]:
        if n.startswith(p):
            return n[len(p):]
    return n


def parse_ids(ids_blob):
    if ids_blob is None or pd.isnull(ids_blob):
        return []
    return [strip_prefix(x) for x in str(ids_blob).split("|")]


def get_id(ids_blob, prefix):
    ids = parse_ids(ids_blob)
    matches = []
    for i in ids:
        if i.startswith(prefix):
            m = re.match(rf"{prefix}\s*(.+)", i)
            if m:
                matches.append(f"{prefix} {m.group(1).strip()}")
    if not matches:
        return None
    matches = sorted(set(matches), key=len)
    return matches[0]


def resolve_one(source_id, simbad):
    q = f"Gaia DR3 {source_id}"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            obj = simbad.query_object(q)
        except Exception:
            obj = None

    if obj is None or len(obj) == 0:
        return dict(source_id=source_id, sao=None, hd=None, tyc=None, spt=None)

    row = obj[0]

    ids = row["IDS"] if "IDS" in obj.colnames else None
    if isinstance(ids, bytes):
        ids = ids.decode()

    spt = row["SP_TYPE"] if "SP_TYPE" in obj.colnames else None
    if isinstance(spt, bytes):
        spt = spt.decode()

    return dict(
        source_id=source_id,
        sao=get_id(ids, "SAO"),
        hd=get_id(ids, "HD"),
        tyc=get_id(ids, "TYC"),
        spt=(spt.strip() if spt else ""),
    )


def load_cache(path):
    expected_cols = ["source_id", "sao", "hd", "tyc", "spt"]
    if path.exists():
        df = normalize_column_names(pd.read_csv(path))
        for col in expected_cols:
            if col not in df.columns:
                df[col] = np.nan
        return df[expected_cols].copy()
    return pd.DataFrame(columns=expected_cols)


def save_cache(df, path):
    df.to_csv(path, index=False)


def resolve_all(source_ids, cache_file, sleep_s=0.1, max_resolve=None):
    cache = load_cache(cache_file)

    done = set(pd.to_numeric(cache["source_id"], errors="coerce").dropna().astype(int))
    missing = [s for s in source_ids if s not in done]

    if max_resolve is not None:
        missing = missing[:max_resolve]

    simbad = make_simbad()
    rows = []

    for i, sid in enumerate(missing, 1):
        rows.append(resolve_one(int(sid), simbad))
        if i % 50 == 0:
            print(f"{i}/{len(missing)} resolved")
        time.sleep(sleep_s)

    if rows:
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        cache = cache.drop_duplicates("source_id", keep="last")
        save_cache(cache, cache_file)

    return cache


def deg_to_hms(ra):
    h = int(ra / 15)
    m = int((ra / 15 - h) * 60)
    s = ((ra / 15 - h) * 60 - m) * 60
    return h, m, s


def deg_to_dms(dec):
    sign = "+" if dec >= 0 else "-"
    x = abs(dec)
    d = int(x)
    m = int((x - d) * 60)
    s = ((x - d) * 60 - m) * 60
    return sign, d, m, s


def select(df, gmax, cut):
    return df[(df.phot_g_mean_mag < gmax) & (df.slope_absdiff <= cut)].copy()


def format_table(df, names):
    df = df.merge(names, on="source_id", how="left")

    ra = df.ra.apply(deg_to_hms)
    dec = df.dec.apply(deg_to_dms)

    df["RAh"] = [x[0] for x in ra]
    df["RAm"] = [x[1] for x in ra]
    df["RAs"] = [x[2] for x in ra]

    df["DEsign"] = [x[0] for x in dec]
    df["DEd"] = [x[1] for x in dec]
    df["DEm"] = [x[2] for x in dec]
    df["DEs"] = [x[3] for x in dec]

    df["sao"] = df["sao"].fillna("")
    df["hd"] = df["hd"].fillna("")
    df["tyc"] = df["tyc"].fillna("")
    df["spt"] = df["spt"].fillna("")

    # Pipeline defines slope as (F2-F1)/(λ2-λ1) with λ in nm; per µm multiply by 1000.
    df["slope_diff_per_um"] = pd.to_numeric(df["slope_absdiff"], errors="coerce") * 1000.0

    return df


def make_csv_table(full):
    return full[
        [
            "source_id",
            "sao",
            "hd",
            "tyc",
            "spt",
            "ra",
            "dec",
            "phot_g_mean_mag",
            "slope_diff_per_um",
        ]
    ].rename(
        columns={
            "source_id": "GaiaDR3",
            "sao": "SAO",
            "hd": "HD",
            "tyc": "TYC",
            "spt": "SpT",
            "ra": "RAd",
            "dec": "DEd_deg",
            "phot_g_mean_mag": "Gmag",
            "slope_diff_per_um": "SlopeDiffperum",
        }
    )


def write_header(f, title: str) -> None:
    f.write("Title: Gaia XP Solar Analog Catalog\n")
    f.write("Authors: Ye Q.\n")
    f.write(f"Table: {title}\n")
    f.write("=" * 80 + "\n")
    f.write("Byte-by-byte Description of file:\n")
    f.write("-" * 80 + "\n")
    f.write("   Bytes Format Units    Label     Explanations\n")
    f.write("-" * 80 + "\n")
    f.write("   1- 19 I19    ---      GaiaDR3   Gaia DR3 source identifier\n")
    f.write("  21- 32 A12    ---      SAO       SAO identifier (if available)\n")
    f.write("  34- 45 A12    ---      HD        HD identifier (if available)\n")
    f.write("  47- 62 A16    ---      TYC       TYC identifier (if available)\n")
    f.write("  64- 73 A10    ---      SpT       Spectral type\n")
    f.write("  75- 76 I2     h        RAh       Hour of Right Ascension (J2000)\n")
    f.write("  78- 79 I2     min      RAm       Minute of Right Ascension (J2000)\n")
    f.write("  81- 85 F5.2   s        RAs       Second of Right Ascension (J2000)\n")
    f.write("  87- 87 A1     ---      DE-       Sign of Declination (J2000)\n")
    f.write("  88- 89 I2     deg      DEd       Degree of Declination (J2000)\n")
    f.write("  91- 92 I2     arcmin   DEm       Arcminute of Declination (J2000)\n")
    f.write("  94- 97 F4.1   arcsec   DEs       Arcsecond of Declination (J2000)\n")
    f.write("  99-109 F11.6  deg      RAd       Right Ascension (deg, J2000)\n")
    f.write(" 111-121 F11.6  deg      DEd_deg   Declination (deg, J2000)\n")
    f.write(" 123-128 F6.3   mag      Gmag      Gaia G-band magnitude\n")
    f.write(
        " 130-145 E12.6  1/um     SlopeDiffperum  "
        "|dF/dλ* - dF/dλ_sun| for normalized XP flux; slopes per micrometer; "
        "see pipeline band 450-900 nm.\n"
    )
    f.write("-" * 80 + "\n")


def write_txt(df, path, title):
    with open(path, "w", encoding="utf-8") as f:
        write_header(f, title)

        header = (
            f"{'GaiaDR3':>19s} "
            f"{'SAO':<12s} "
            f"{'HD':<12s} "
            f"{'TYC':<16s} "
            f"{'SpT':<10s} "
            f"{'RAh':>2s} "
            f"{'RAm':>2s} "
            f"{'RAs':>5s} "
            f"{'S':>1s} "
            f"{'DEd':>2s} "
            f"{'DEm':>2s} "
            f"{'DEs':>4s} "
            f"{'RAd':>11s} "
            f"{'DEd_deg':>11s} "
            f"{'Gmag':>6s} "
            f"{'Slope/um':>12s}"
        )
        f.write(header + "\n")

        for _, r in df.iterrows():
            f.write(
                f"{int(r.source_id):19d} "
                f"{str(r.sao):<12.12s} "
                f"{str(r.hd):<12.12s} "
                f"{str(r.tyc):<16.16s} "
                f"{str(r.spt):<10.10s} "
                f"{int(r.RAh):02d} "
                f"{int(r.RAm):02d} "
                f"{float(r.RAs):05.2f} "
                f"{str(r.DEsign):>1s} "
                f"{int(r.DEd):02d} "
                f"{int(r.DEm):02d} "
                f"{float(r.DEs):04.1f} "
                f"{float(r.ra):11.6f} "
                f"{float(r.dec):11.6f} "
                f"{float(r.phot_g_mean_mag):6.3f} "
                f"{float(r.slope_diff_per_um):12.5e}\n"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--outdir", default="machine_readable_catalogs")
    ap.add_argument("--cache-file", default="simbad_name_cache.csv")
    ap.add_argument("--sleep-s", type=float, default=0.1)
    ap.add_argument("--max-resolve", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(exist_ok=True)

    df = normalize_column_names(pd.read_csv(args.catalog))

    _, _, cut1 = slope_thresholds_from_windows(CONTINUUM_WINDOWS_NM)

    # Only 1% slope-equivalent selection (slope_absdiff <= cut1).
    sets = {
        "glt12_1": (select(df, 12, cut1), "Solar analog catalog: G<12 and 1% slope-equivalent"),
        "glt10_1": (select(df, 10, cut1), "Solar analog catalog: G<10 and 1% slope-equivalent"),
        "glt8_1": (select(df, 8, cut1), "Solar analog catalog: G<8 and 1% slope-equivalent"),
    }

    ids = sorted(set(pd.concat([s.source_id for s, _ in sets.values()])))
    names = resolve_all(ids, Path(args.cache_file), sleep_s=args.sleep_s, max_resolve=args.max_resolve)

    for key, (sub, title) in sets.items():
        full = format_table(sub, names)
        csv = make_csv_table(full)

        csv.to_csv(out / f"{key}.csv", index=False)
        write_txt(full, out / f"{key}.txt", title)

        print(key, len(full))


if __name__ == "__main__":
    main()
