from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly: `python PC.py` from within `src/model/PC/`.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(_PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(_PROJECT_ROOT))
from src.model.util.dataset import DatasetConfig
from src.model.Predictive_Coding.pc_network import PCNetworkConfig
from src.model.Predictive_Coding.trainer import PCTrainer, TrainerConfig

def main() -> None:
	### Config ###
	ds_cfg = DatasetConfig(
		train_start_year=2014,
		train_end_year=2023,
		test_year=2024,
	)

	pc_cfg = PCNetworkConfig(
		hidden_activation="tanh", # tanh
		output_activation="sigmoid",
		learning_rate=1e-2, # 1e-2
		inference_lr=2e-1,  # 2e-1
		inference_steps=20, # 20
		random_seed=42,
	)

	trainer_cfg = TrainerConfig(
		epochs=50, # converge at 50
		batch_size=256,
		threshold=0.5,
		print_every=1,
	)

	trainer = PCTrainer(
		pc_cfg=pc_cfg,
		trainer_cfg=trainer_cfg,
		ds_cfg=ds_cfg,
		hidden_sizes=(64, 8),
	)
	print(f"X_train shape: {trainer.X_train.shape}, X_test shape: {trainer.X_test.shape}")
	### ### ###

	metrics = trainer.fit()

	print("\nFinal metrics")
	for split in ("train", "test"):
		print(f"\n{split.upper()}")
		for k, v in metrics[split].items():
			print(f"- {k:>16}: {v:.5f}")

if __name__ == "__main__":
	main()

