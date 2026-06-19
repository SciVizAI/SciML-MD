# tests/

**Validation and testing suite for SciML-MD.**

## Test files

| File | What it validates |
|---|---|
| `test_integration.py` | End-to-end pipeline regression |
| `test_signals.py` | Rarity, transition surprise, local density unit tests |
| `test_reproducibility.py` | Deterministic output across runs |
| `test_scientific_validation.py` | Chapman-Kolmogorov, implied timescales |
| `test_statistical_validation.py` | Bootstrap confidence intervals |
| `test_dataset_validation.py` | Trajectory and topology integrity |
| `test_pipeline_edge_cases.py` | Short trajectories, disconnected states |
| `test_chapter9_evaluation.py` | Thesis Chapter 9 results validation |
| `run_all_validation.py` | Runs all tests, saves to TEST_RESULTS.md |

## Run all tests
```bash
python tests/run_all_validation.py
```

## Run individual test
```bash
pytest tests/test_signals.py -v
```
