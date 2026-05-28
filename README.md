# Tennis-match-prediction

## New Model

The pipeline now supports a hidden Markov model classifier via `--model hmm`.

Install its dependency before running:

```bash
python -m pip install hmmlearn
```

Example:

```bash
python main.py --model hmm --optimizer optuna --validation walk_forward --n_trials 100 --n_jobs 4
```

HMM runs on CPU in this pipeline.

Resume Optuna runs:

If an Optuna study is interrupted you can resume it by passing `--resume` when running the pipeline. The HMM study is stored as an SQLite DB in the model output folder (e.g. `outputs/hmm/.../hmm_optuna.db`). Example resume command:

```bash
python main.py --model hmm --optimizer optuna --validation walk_forward --n_trials 100 --n_jobs 4 --resume
```

Notes:
- When `--resume` is set the pipeline will load the existing Optuna study and continue until the requested total `--n_trials` is reached.
- If `hmm_model.joblib` and `hmm_scaler.joblib` already exist, passing `--resume` will still continue HPO; omit `--resume` to return the saved model immediately.
- For reliability with SQLite, consider `--n_jobs 1` when running many parallel workers to avoid DB lock issues.