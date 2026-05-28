from dataclasses import dataclass


@dataclass(frozen=True)
class HMMTuningConfig:
    n_components_min: int = 2
    n_components_max: int = 6
    n_iter_min: int = 50
    n_iter_max: int = 200
    seq_len_min: int = 3
    seq_len_max: int = 10
    covariance_types: tuple[str, ...] = ("diag", "spherical")
    tol_min: float = 1e-4
    tol_max: float = 1e-2
    min_covar_min: float = 1e-4
    min_covar_max: float = 1e-2


HMM_TUNING = HMMTuningConfig()
HMM_DEFAULT_N_TRIALS = 50
HMM_PRUNER_STARTUP_TRIALS = 10
HMM_EARLY_STOP_PATIENCE = 50
HMM_MAX_EPOCHS = 1000