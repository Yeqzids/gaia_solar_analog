# Gaia solar analog catalog

Utilities for building and validating an all-sky Gaia DR3 solar-analog catalog using [Gaia@AIP](https://gaia.aip.de/), optional empirical XP correction following [Huang et al. (2024)](https://iopscience.iop.org/article/10.3847/1538-4365/ad18b1), literature cross-checks, and publication tables.

## Dependencies

Install Python 3.10+ and:

```bash
pip install gaiaxpy pyvo requests pandas numpy scipy pyarrow astropy matplotlib astroquery
```

Then, the [Gaia XP empirical correction](https://iopscience.iop.org/article/10.3847/1538-4365/ad18b1) code. In `gaia_stage2_aip_xpcorrect.py`, set `LOCAL_GAIAXP_CORRECTION_PATH` to the directory from which `from GaiaDR3XPspectracorrectionV1 import Gaia_Correction_V1` works.

## What is in this repository

| File | Role |
|------|------|
| `gaia_stage1_aip.py` | Stage-1 candidate selection via Gaia@AIP TAP. |
| `gaia_stage2_aip_xpcorrect.py` | Stage-2 pipeline with Huang et al.–style XP correction and catalog assembly. |
| `plot_sky_density.py` | Sky density maps (e.g. Aitoff, radius-count mode). |
| `literature_benchmark.py` | Resolve literature names (SIMBAD), compare to the ranked catalog, optional spectra plots. |
| `make_tables.py` | Machine-readable tables from the final catalog. |
| `solar_analogs_final.txt` | Combined list of solar analogs in [Farnham et al. 2000](https://ui.adsabs.harvard.edu/abs/2000Icar..147..180F), [Giribaldi et al. 2019](https://ui.adsabs.harvard.edu/abs/2019A&A...629A..33G), [Lewin et al. 2020](https://ui.adsabs.harvard.edu/abs/2020AJ....160..130L), and [Marsset et al. 2020](https://ui.adsabs.harvard.edu/abs/2020ApJS..247...73M). |
| `solar_reference.csv` | [ASTM G-173 extraterrestrial solar spectrum](https://store.astm.org/g0173-23.html). |

## Workflow

### 1. Stage 1 — Gaia@AIP candidates

Query Gaia@AIP and write Stage-1 candidate CSVs (see script help for paths and options):

```bash
python gaia_stage1_aip.py --aip-token "$GAIA_AIP_TOKEN"
```

### 2. Stage 2 — Empirical XP correction (Huang et al. 2024)

After Stage 1, calibrate and correct spectra, then build the ranked catalog. Adjust `--outdir`, `--batch-size`, and paths to match your layout:

```bash
python gaia_stage2_aip_xpcorrect.py \
  --stage1-csv gaia_solar_analogs_out/gaia_stage1_candidates.csv \
  --solar-csv solar_reference.csv \
  --outdir gaia_solar_analogs_out_xpcorrect \
  --aip-token "$GAIA_AIP_TOKEN" \
  --batch-size 1000
```

### 3. Produce catalog tables at 1 % slope-equivalent

```bash
python make_tables.py \
  --catalog gaia_solar_analogs_out_g12_xpcorrect/gaia_allsky_solar_analog_catalog.csv \
  --outdir machine_readable_tables_g12_xpcorrect
```

### 4. Sky density plot

Example: density of sources passing a slope-quality cut, 7° neighbour count, 1° map pixels, PNG plus PDF:

```bash
python plot_sky_density.py \
  --input slope_quality_products/catalog_slope_1pct.csv \
  --output slope_quality_products/catalog_slope_1pct_within7deg.png \
  --mode radius-count \
  --radius-deg 7 \
  --map-step-deg 1 \
  --also-pdf
```

### 5. Literature benchmark

Compare the final catalog to a curated name list and optionally plot literature spectra vs. the solar reference:

```bash
python literature_benchmark.py \
  --catalog gaia_solar_analogs_out_g12_xpcorrect/gaia_allsky_solar_analog_catalog.csv \
  --names solar_analogs_final.txt \
  --outdir solar_analogs_final_benchmark \
  --solar-csv solar_reference.csv \
  --xpdir gaia_solar_analogs_out_g12_xpcorrect/xp_batches \
  --plot
```

To restrict metrics to a wavelength interval and relax/tighten the slope gate (example: 550–850 nm, 5 % slope-equivalent):

```bash
python literature_benchmark.py \
  --catalog gaia_solar_analogs_out_g12_xpcorrect/gaia_allsky_solar_analog_catalog.csv \
  --names solar_analogs_final.txt \
  --outdir solar_analogs_final_benchmark_wl550_850 \
  --solar-csv solar_reference.csv \
  --xpdir gaia_solar_analogs_out_g12_xpcorrect/xp_batches \
  --wavelength-range 550 850 \
  --slope-equivalent-percent 5
```

## Reference

This effort made uses of the Gaia DR3 data releaseas well as the [Huang et al. (2024) correction model](https://iopscience.iop.org/article/10.3847/1538-4365/ad18b1). A paper is being prepared.
