# Paired Discordance Certificates for Frozen Diagnostic AI Policies

Reproducible implementation of a finite-sample audit that compares frozen diagnostic policies across disease prevalence, false-positive harm, and bounded within-class (total-variation) population shift.

For two policies with frozen score cutoffs, the net-benefit difference is linear in two paired class-conditional differences (true-positive and false-positive rates). Bounding those two differences therefore bounds the comparison over the entire prevalence–harm plane at once. Each paired difference is bounded through its discordance rate and the direction of discordance, both covered by exact Clopper–Pearson intervals, so precision is governed by how often the models disagree. The audit returns supported, unresolved, and transport-sensitive comparisons, a closed-form superiority frontier, and a sufficient conditional-shift reserve.

## Contents

| Path | Description |
|---|---|
| `code/paired_certificate.py` | Paired certificate, DKW baseline, superiority maps, transport reserve, simulation study, planning table, figures |
| `data/` | Frozen audit predictions and model metrics for Breast Cancer Wisconsin (Diagnostic) and Statlog (Heart) |
| `notebooks/diagnostic_model_comparison.ipynb` | Executed notebook: data splits, model training, cutoff selection, DKW analysis, bootstrap sensitivity, exact TV-shift experiment |
| `results/dgpc_results.json` | Complete numerical output of `paired_certificate.py` |
| `figures/` | Figures produced by the code |

## Reproduction

```bash
pip install -r requirements.txt
python code/paired_certificate.py        # about 10 s on a CPU; rewrites results/ and figures/
```

The notebook downloads Statlog (Heart) from the UCI repository and loads Breast Cancer Wisconsin through scikit-learn. All random seeds are fixed (`20261017`).

## Data

Both datasets are public de-identified benchmarks from the UCI Machine Learning Repository. They are small and historical; results are a methods demonstration, not clinical validation.

## Citation

See `CITATION.cff`. Please also cite the accompanying article once published.

## License

MIT (code). The UCI datasets remain under their original terms.
