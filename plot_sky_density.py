#!/usr/bin/env python3
"""
plot_sky_density.py

Create a sky-density plot (Aitoff projection) from a CSV containing
`ra` and `dec` columns in degrees.

Example
-------
python3 plot_sky_density.py \\
  --input gaia_allsky_solar_analog_catalog.csv \\
  --output sky_density.png \\
  --also-pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import BarycentricTrueEcliptic
from astropy.coordinates import Galactic
from astropy.coordinates import SkyCoord

FULL_SKY_AREA_DEG2 = 4.0 * np.pi * (180.0 / np.pi) ** 2  # ~41252.96

# Embedded bitmap resolution for PDF: dense hexbin/pcolormesh are rasterized so
# vector export stays small and avoids moiré from millions of tiny paths.
PDF_RASTER_DPI = 300


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def ra_deg_to_aitoff_rad(ra_deg: np.ndarray) -> np.ndarray:
    # Aitoff expects longitudes in [-pi, pi]. Convert RA from [0, 360) to [-180, 180).
    wrapped_deg = ((ra_deg + 180.0) % 360.0) - 180.0
    return np.deg2rad(wrapped_deg)


def make_plot(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    output_path: Path,
    gridsize_x: int,
    min_count: int,
    cmap: str,
    title: str,
    pdf_path: Optional[Path] = None,
) -> None:
    x = ra_deg_to_aitoff_rad(ra_deg)
    y = np.deg2rad(dec_deg)

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111, projection="aitoff")

    hb = ax.hexbin(
        x,
        y,
        gridsize=gridsize_x,
        mincnt=min_count,
        cmap=cmap,
        bins="log",
        rasterized=True,
    )
    cb = fig.colorbar(hb, ax=ax, pad=0.08)
    cb.set_label("log10(count per hexbin)")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.6, linestyle="--", color="black")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    if pdf_path is not None:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, format="pdf", dpi=PDF_RASTER_DPI)
    plt.close(fig)


def unit_vectors_from_radec_deg(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    cosd = np.cos(dec)
    x = cosd * np.cos(ra)
    y = cosd * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x, y, z])


def make_radius_count_map(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    radius_deg: float,
    map_step_deg: float,
    pixel_batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if radius_deg <= 0:
        raise ValueError("--radius-deg must be > 0")
    if map_step_deg <= 0:
        raise ValueError("--map-step-deg must be > 0")
    if pixel_batch_size <= 0:
        raise ValueError("--pixel-batch-size must be > 0")

    lon_centers_deg = np.arange(-180.0, 180.0 + 0.5 * map_step_deg, map_step_deg)
    lat_centers_deg = np.arange(-90.0, 90.0 + 0.5 * map_step_deg, map_step_deg)
    lon_grid_deg, lat_grid_deg = np.meshgrid(lon_centers_deg, lat_centers_deg)

    # Convert map longitudes (-180..180) to RA convention (0..360) for spherical distances.
    ra_grid_deg = (lon_grid_deg + 360.0) % 360.0
    pix_vec = unit_vectors_from_radec_deg(ra_grid_deg.ravel(), lat_grid_deg.ravel())
    src_vec = unit_vectors_from_radec_deg(ra_deg, dec_deg)
    cos_thresh = np.cos(np.deg2rad(radius_deg))

    counts_flat = np.zeros(pix_vec.shape[0], dtype=np.int32)
    src_t = src_vec.T
    for i in range(0, pix_vec.shape[0], pixel_batch_size):
        j = min(i + pixel_batch_size, pix_vec.shape[0])
        dots = pix_vec[i:j] @ src_t
        counts_flat[i:j] = np.sum(dots >= cos_thresh, axis=1, dtype=np.int32)

    counts = counts_flat.reshape(lon_grid_deg.shape)
    return lon_grid_deg, lat_grid_deg, counts


def make_radius_plot(
    lon_grid_deg: np.ndarray,
    lat_grid_deg: np.ndarray,
    counts: np.ndarray,
    output_path: Path,
    cmap: str,
    title: str,
    log_scale: bool,
    radius_deg: float,
    show_ecliptic: bool,
    show_galactic: bool,
    pdf_path: Optional[Path] = None,
) -> None:
    def split_on_wrap(x_rad: np.ndarray, y_rad: np.ndarray, jump_deg: float = 180.0) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x_rad, dtype=float).copy()
        y = np.asarray(y_rad, dtype=float).copy()
        if len(x) < 2:
            return x, y
        jump = np.abs(np.diff(x))
        cut_idx = np.where(jump > np.deg2rad(jump_deg))[0]
        for k in cut_idx[::-1]:
            x = np.insert(x, k + 1, np.nan)
            y = np.insert(y, k + 1, np.nan)
        return x, y

    # Flip longitude for display so RA decreases to the right.
    lon = np.deg2rad(-lon_grid_deg)
    lat = np.deg2rad(lat_grid_deg)
    values = np.log10(np.maximum(counts, 1)) if log_scale else counts.astype(float)

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111, projection="aitoff")
    cmap_obj = plt.get_cmap(cmap).copy()
    # Force a sharp 0->1 visual break: count==0 uses the "under" color.
    cmap_obj.set_under("black")
    if log_scale:
        # log10(1)=0 starts normal scale; zero-count pixels are set below vmin.
        values = np.where(counts > 0, values, -1.0)
        mesh = ax.pcolormesh(
            lon, lat, values, shading="auto", cmap=cmap_obj, vmin=0.0, rasterized=True
        )
    else:
        mesh = ax.pcolormesh(
            lon, lat, values, shading="auto", cmap=cmap_obj, vmin=0.5, rasterized=True
        )
    cb = fig.colorbar(mesh, ax=ax, pad=0.08)
    cb.set_label(rf"N stars within ${radius_deg:g}^\circ$ radius")
    if title:
        ax.set_title(title)

    # Set tick positions directly in displayed longitude space, then convert
    # those positions back to RA-hour labels consistent with the flipped axis.
    tick_pos_deg = np.arange(-150.0, 151.0, 30.0)
    ax.set_xticks(np.deg2rad(tick_pos_deg))
    tick_labels = []
    for xdeg in tick_pos_deg:
        lon_deg = -xdeg  # undo display flip
        ra_deg = lon_deg % 360.0
        rah = int(round(ra_deg / 15.0)) % 24
        tick_labels.append(f"{rah}h")
    ax.set_xticklabels(tick_labels)
    ax.grid(True, alpha=0.6, linestyle="--", color="black")

    if show_ecliptic:
        ecl_lon = np.linspace(0.0, 360.0, 721)
        ecl_lat = np.zeros_like(ecl_lon)
        ecl_icrs = SkyCoord(
            lon=ecl_lon * u.deg,
            lat=ecl_lat * u.deg,
            frame=BarycentricTrueEcliptic,
        ).icrs
        ecl_x_deg = -(((ecl_icrs.ra.deg + 180.0) % 360.0) - 180.0)
        ecl_y_deg = ecl_icrs.dec.deg
        ecl_x_rad, ecl_y_rad = split_on_wrap(np.deg2rad(ecl_x_deg), np.deg2rad(ecl_y_deg))
        ax.plot(
            ecl_x_rad,
            ecl_y_rad,
            linestyle="--",
            linewidth=1.3,
            color="orange",
            alpha=0.9,
            label="Ecliptic",
        )

    if show_galactic:
        gal_lon = np.linspace(0.0, 360.0, 721)
        gal_lat = np.zeros_like(gal_lon)
        gal_icrs = SkyCoord(
            l=gal_lon * u.deg,
            b=gal_lat * u.deg,
            frame=Galactic,
        ).icrs
        gal_x_deg = -(((gal_icrs.ra.deg + 180.0) % 360.0) - 180.0)
        gal_y_deg = gal_icrs.dec.deg
        gal_x_rad, gal_y_rad = split_on_wrap(np.deg2rad(gal_x_deg), np.deg2rad(gal_y_deg))
        ax.plot(
            gal_x_rad,
            gal_y_rad,
            linestyle="--",
            linewidth=1.3,
            color="red",
            alpha=0.9,
            label="Galactic plane",
        )

    if show_ecliptic or show_galactic:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=2,
            frameon=True,
            fontsize=9,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    if pdf_path is not None:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, format="pdf", dpi=PDF_RASTER_DPI)
    plt.close(fig)


def gridsize_for_hex_area_deg2(hex_area_deg2: float) -> int:
    """
    Approximate matplotlib hexbin gridsize for a target sky area per hex.
    For a single integer gridsize=n, matplotlib uses roughly ~0.5*n^2 hexes.
    """
    if hex_area_deg2 <= 0:
        raise ValueError("--hex-area-deg2 must be > 0")
    n_hex = FULL_SKY_AREA_DEG2 / hex_area_deg2
    gridsize_x = int(max(10, round(np.sqrt(2.0 * n_hex))))
    return gridsize_x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a sky-density plot from CSV ra/dec columns")
    p.add_argument("--input", required=True, help="Input CSV file with ra and dec columns (degrees)")
    p.add_argument("--output", default="sky_density.png", help="Output raster path (e.g. PNG)")
    p.add_argument(
        "--pdf",
        default=None,
        help="Also save a vector PDF to this path",
    )
    p.add_argument(
        "--also-pdf",
        action="store_true",
        help="Also save a PDF beside the raster (same basename, .pdf extension)",
    )
    p.add_argument(
        "--mode",
        choices=["hexbin", "radius-count"],
        default="radius-count",
        help="Plot mode: hexbin density or count-within-radius map",
    )
    p.add_argument(
        "--hex-area-deg2",
        type=float,
        default=1.0,
        help="Target sky area per hexbin in square degrees (approximate)",
    )
    p.add_argument(
        "--gridsize",
        type=int,
        default=None,
        help="Override hexbin gridsize directly (if set, ignores --hex-area-deg2)",
    )
    p.add_argument("--min-count", type=int, default=1, help="Minimum sources per hexbin to draw")
    p.add_argument("--radius-deg", type=float, default=5.0, help="Radius in degrees for radius-count mode")
    p.add_argument("--map-step-deg", type=float, default=1.0, help="Map grid spacing in degrees for radius-count mode")
    p.add_argument("--pixel-batch-size", type=int, default=256, help="Number of map pixels per compute batch")
    p.add_argument("--log-scale", action="store_true", help="Use log10 color scale in radius-count mode")
    p.add_argument("--no-ecliptic", action="store_true", help="Do not draw dashed ecliptic line")
    p.add_argument("--no-galactic", action="store_true", help="Do not draw dashed galactic plane line")
    p.add_argument("--cmap", default="viridis", help="Matplotlib colormap")
    p.add_argument("--title", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = normalize_column_names(pd.read_csv(args.input))

    if "ra" not in df.columns or "dec" not in df.columns:
        raise ValueError("Input CSV must contain columns named 'ra' and 'dec'.")

    ra = pd.to_numeric(df["ra"], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(df["dec"], errors="coerce").to_numpy(dtype=float)

    good = np.isfinite(ra) & np.isfinite(dec) & (dec >= -90.0) & (dec <= 90.0)
    n_total = len(df)
    n_used = int(np.sum(good))
    if n_used == 0:
        raise ValueError("No valid ra/dec rows found after filtering.")

    pdf_out: Optional[Path] = None
    if args.pdf:
        pdf_out = Path(args.pdf)
    elif args.also_pdf:
        pdf_out = Path(args.output).with_suffix(".pdf")

    if args.mode == "hexbin":
        gridsize_x = args.gridsize if args.gridsize is not None else gridsize_for_hex_area_deg2(args.hex_area_deg2)
        make_plot(
            ra_deg=ra[good],
            dec_deg=dec[good],
            output_path=Path(args.output),
            gridsize_x=gridsize_x,
            min_count=args.min_count,
            cmap=args.cmap,
            title=args.title,
            pdf_path=pdf_out,
        )
        print(f"Read {n_total} rows; plotted {n_used} valid sky positions.")
        if args.gridsize is None:
            print(f"Used approximate 1 hex = {args.hex_area_deg2:g} deg^2 (gridsize={gridsize_x}).")
        else:
            print(f"Used manual gridsize={gridsize_x}.")
    else:
        lon_grid_deg, lat_grid_deg, counts = make_radius_count_map(
            ra_deg=ra[good],
            dec_deg=dec[good],
            radius_deg=args.radius_deg,
            map_step_deg=args.map_step_deg,
            pixel_batch_size=args.pixel_batch_size,
        )
        make_radius_plot(
            lon_grid_deg=lon_grid_deg,
            lat_grid_deg=lat_grid_deg,
            counts=counts,
            output_path=Path(args.output),
            cmap=args.cmap,
            title=args.title,
            log_scale=args.log_scale,
            radius_deg=args.radius_deg,
            show_ecliptic=(not args.no_ecliptic),
            show_galactic=(not args.no_galactic),
            pdf_path=pdf_out,
        )
        print(f"Read {n_total} rows; plotted {n_used} valid sky positions.")
        print(
            f"Radius-count mode: radius={args.radius_deg:g} deg, map_step={args.map_step_deg:g} deg, "
            f"pixels={counts.size}."
        )
        n_zero = int(np.sum(counts == 0))
        print(f"Pixels with N=0: {n_zero} / {counts.size}")
    print(f"Wrote: {args.output}")
    if pdf_out is not None:
        print(f"Wrote PDF: {pdf_out}")


if __name__ == "__main__":
    main()
