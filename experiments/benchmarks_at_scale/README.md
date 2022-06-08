# Benchmarks at Scale

## Main results

## Reproducibility

To reproduce the main results you have:
1. Install the conda environment using,

```bash
conda env create -f environment.yml
```

2. Activate the environment using,

```bash
conda activate benchmarks_at_scale
```

3. Generate the data using,

```bash
python -m src.data
```

4. Run the experiments using,

```bash
python -m src.experiment
```
