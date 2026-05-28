import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.paths import ensure_writable_path


def plot_optuna_history(
    study,
    save_path,
    filename="optuna_optimization_history.png",
    title="Optuna Optimization History (Validation Accuracy)",
):
    save_path = ensure_writable_path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    trials = study.trials_dataframe()
    if not trials.empty and "value" in trials.columns:
        sns.lineplot(data=trials, x="number", y="value", marker="o")
        plt.title(title)
        plt.xlabel("Trial Number")
        plt.ylabel("Validation Accuracy")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path / filename, dpi=300)
    plt.close()
