# Extreme Maximum Temperature Events: A Weibull Approach with Adaptive Plotting Positions

This repository contains the data pipeline, analysis code, and manuscript source for the paper:

> Olivera, S. (2026). *Extreme Maximum Temperature Events in Mexico: A Weibull Distribution Approach with Adaptive Plotting Positions*.

---

## Abstract

An adaptive methodology is proposed for modeling the right tail of daily maximum temperature exceedances using a two-parameter Weibull mixture model, extending the weighted least-squares framework of Olivera and Heard (2019) along three dimensions: i) the modal threshold is estimated via kernel density estimation (KDE), ii) the number of bins is determined by the Freedman–Diaconis rule, and iii) the plotting-position parameter *a* is optimized adaptively. The methodology is applied to daily maximum temperature records from five cities in Mexico and the southwestern United States across three twenty-year periods (1940–1968, 1969–1996, 1997–2026). Results show that the modal threshold μ\* exhibits an upward trend in most cities while the shape parameter remains comparatively stable, indicating that changes in thermal extremes are expressed primarily as shifts in typical temperature levels.

---

## Repository structure

├── paper/ LaTeX source (English and Spanish versions)
├── code/ Python analysis and figure scripts
├── figures/ All figures used in the paper
├── results/ Summary table of estimated parameters
└── data/ Excel files used in prueba28.py


## Reproducing the results

### 1. Install dependencies

```bash
pip install -r code/requirements.txt
```


### 2. Run the analysis

```bash
python code/prueba28.py
```

This produces `output/all_results.pkl` and an Excel summary.

### 3. Generate figures

```bash
python code/graficas27.py
```

Figures are saved to `output/figuras_articulo/` and `output/figuras_anexo/`.

---

## Key methodology

| Step | Description |
|------|-------------|
| **1. Modal threshold** | KDE with Gaussian kernel and Scott's bandwidth rule |
| **2. Exceedance variable** | T = max(X − μ\*, 0) |
| **3. Class construction** | Freedman–Diaconis bin width |
| **4. Mixture model** | F_mix = p₀ + (1 − p₀) · F_Weibull |
| **5. WLS estimation** | Bergman weight function |
| **6. Parameter optimization** | Adaptive plotting position + differential evolution |

---

## Results summary

See [`results/parameters_summary.csv`](results/parameters_summary.csv) for
the full table of estimated parameters (μ\*, k, λ, R², KS statistic) and
exceedance probabilities for all city–period combinations.

---

## Citation

If you use this code or methodology, please cite:

```bibtex
@article{olivera2026,
  author  = {Olivera, Sazcha},
  title   = {Extreme Maximum Temperature Events in Mexico and the
             Southwestern United States: A Weibull Distribution Approach
             with Adaptive Plotting Positions},
  year    = {2026},
}
```

---

## Data source

Temperature data: ERA5 hourly data on single levels from 1940 to present.
Variable: 2 m air temperature (`2m_temperature`).

> Hersbach, H. et al. (2023). ERA5 hourly data on single levels from 1940
> to present. Copernicus Climate Change Service (C3S) Climate Data Store
> (CDS). https://doi.org/10.24381/cds.adbb2d47
