import os
import sys
import json
import joblib
import threading
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog

# --- Setup System Path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing.preprocessing import Preprocessing
from main import evaluate_model_bias 
from src.models.svm.svm_pytorch_optuna import PyTorchLinearSVM


class StdoutRedirector:
    """Safely redirects standard output (sys.stdout) streams into a Tkinter text component."""
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        # Schedule the UI update safely on the main thread
        self.text_widget.after(0, self._insert_text, string)

    def _insert_text(self, string):
        self.text_widget.config(state='normal')
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.config(state='disabled')

    def flush(self):
        pass  # Required for file-like interface compliance


class TennisInferenceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎾 2026 Tennis Inference Hub (RF & PyTorch SVM)")
        self.root.geometry("780x750")
        
        # --- Apply Custom Colors & Styling ---
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.style.configure('TFrame', background='#F0F4F8')
        self.style.configure('TLabel', background='#F0F4F8', foreground='#102A43', font=("Helvetica", 10))
        self.style.configure('Header.TLabel', font=("Helvetica", 12, "bold"), foreground='#0A558C')
        
        self.style.configure('Primary.TButton', font=("Helvetica", 10, "bold"), background='#00A388', foreground='white', padding=6)
        self.style.map('Primary.TButton', background=[('active', '#008C75'), ('disabled', '#BCCCDC')])
        
        self.style.configure('Secondary.TButton', font=("Helvetica", 10, "bold"), background='#2684FF', foreground='white', padding=6)
        self.style.map('Secondary.TButton', background=[('active', '#0052CC'), ('disabled', '#BCCCDC')])
        
        self.style.configure('Warning.TButton', font=("Helvetica", 10, "bold"), background='#D9534F', foreground='white', padding=6)
        self.style.map('Warning.TButton', background=[('active', '#C9302C'), ('disabled', '#BCCCDC')])

        # State Variables
        self.model_dir = None
        self.model_type = None 
        self.data_loaded = False
        
        self.df_2026_raw = None 
        self.X_2026 = None
        self.y_2026 = None
        
        # Artifacts
        self.model = None
        self.scaler = None
        self.pca = None
        self.kmeans = None
        self.svm_poly = None
        self.svm_selector = None
        self.svm_scaler_poly = None
        self.svm_dist_scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = None
        self.expected_base_features = None

        self._build_ui()
        
        # --- Redirect Stdout to UI Console Tool ---
        sys.stdout = StdoutRedirector(self.result_text)

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Section 1: Model Selection ---
        ttk.Label(main_frame, text="1. Select Model Configuration", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 5))
        model_frame = ttk.Frame(main_frame)
        model_frame.pack(fill=tk.X, pady=5)
        
        self.btn_browse = ttk.Button(model_frame, text="📁 Browse Model Folder", style='Secondary.TButton', command=self.browse_model)
        self.btn_browse.pack(side=tk.LEFT, padx=(0, 10))
        self.lbl_model_path = ttk.Label(model_frame, text="No model folder selected...", font=("Helvetica", 9, "italic"), foreground="#627D98")
        self.lbl_model_path.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        # --- Section 2: Loading Status ---
        ttk.Label(main_frame, text="2. Initialize Pipeline", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 5))
        self.btn_load = ttk.Button(main_frame, text="Load Data & Selected Model", style='Primary.TButton', command=self.start_loading_thread)
        self.btn_load.pack(fill=tk.X, pady=5)
        self.lbl_status = ttk.Label(main_frame, text="Status: Waiting for model selection...", font=("Helvetica", 10, "bold"), foreground="#D9534F")
        self.lbl_status.pack(pady=5)

        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        # --- Section 3: Standard Inference ---
        ttk.Label(main_frame, text="3. Standard Inference & Evaluation", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 5))
        select_frame = ttk.Frame(main_frame)
        select_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(select_frame, text="Select Match:").pack(side=tk.LEFT, padx=(0, 10))
        self.combo_matches = ttk.Combobox(select_frame, state="disabled", width=50)
        self.combo_matches.pack(side=tk.LEFT, fill=tk.X, expand=True)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        self.btn_predict = ttk.Button(btn_frame, text="Predict Selected Match", style='Secondary.TButton', command=self.predict_single_match, state="disabled")
        self.btn_predict.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.btn_eval = ttk.Button(btn_frame, text="Run Full 2026 Batch", style='Secondary.TButton', command=self.run_full_evaluation, state="disabled")
        self.btn_eval.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        # --- Section 4: Upset Match Miner ---
        ttk.Label(main_frame, text="4. Dynamic Upset Analysis (Lower Elo Wins)", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 5))
        upset_frame = ttk.Frame(main_frame)
        upset_frame.pack(fill=tk.X, pady=5)

        ttk.Label(upset_frame, text="Max Upsets to Test:").pack(side=tk.LEFT, padx=(0, 5))
        self.txt_upset_val = ttk.Entry(upset_frame, width=8)
        self.txt_upset_val.insert(0, "10")  # Default value
        self.txt_upset_val.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_miner = ttk.Button(upset_frame, text="🔥 Find & Predict Upsets", style='Warning.TButton', command=self.analyze_upsets, state="disabled")
        self.btn_miner.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # --- Console Output Display ---
        self.result_text = scrolledtext.ScrolledText(main_frame, height=14, state='disabled', font=("Consolas", 10), bg="#1E2A38", fg="#E0E0E0")
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def log_message(self, message):
        # Standard print now goes automatically to UI via redirection, 
        # but we preserve this for backward compatibility or explicit lines.
        print(message)

    def browse_model(self):
        initial_dir = os.path.join(project_root, "weights")
        if not os.path.exists(initial_dir): initial_dir = project_root

        selected_dir = filedialog.askdirectory(initialdir=initial_dir, title="Select Model Artifact Folder")
        if selected_dir:
            self.model_dir = Path(selected_dir)
            if list(self.model_dir.glob("*.pth")):
                self.model_type, type_str = "SVM", "PyTorch SVM"
            elif list(self.model_dir.glob("*_config.json")):
                self.model_type, type_str = "RF", "Random Forest"
            else:
                messagebox.showwarning("Invalid Folder", "Could not detect a valid model in this directory.")
                self.model_dir = None
                return
                
            self.lbl_model_path.config(text=f"[{type_str}] .../{self.model_dir.parent.name}/{self.model_dir.name}", foreground="#00A388")
            self.lbl_status.config(text=f"Status: Ready to load {type_str} pipeline.", foreground="#102A43")

    def start_loading_thread(self):
        if not self.model_dir: return
        self.btn_load.config(state="disabled")
        self.btn_browse.config(state="disabled")
        self.lbl_status.config(text="Status: Processing Data & Loading Weights...", foreground="#D9534F")
        threading.Thread(target=self.load_data_and_models, daemon=True).start()

    def load_data_and_models(self):
        try:
            prep = Preprocessing()
            raw_data = prep._load()
            raw_data['year'] = pd.to_datetime(raw_data['tourney_date'], format='%Y%m%d', errors='coerce').dt.year
            
            matches_up_to_2024 = raw_data[raw_data['year'] <= 2024]
            frozen_ratio = int(len(matches_up_to_2024) * 0.90) / len(raw_data)
            
            print("Running Preprocessing Pipeline...")
            data = prep.run(train_ratio=frozen_ratio)

            df_2026 = data[data['year'] == 2026].copy()
            if 'is_augmented' in df_2026.columns:
                df_2026 = df_2026[df_2026['is_augmented'] == 0].drop(columns=['is_augmented'])

            try:
                self.df_2026_raw = raw_data.loc[df_2026.index].copy()
            except KeyError:
                self.df_2026_raw = raw_data[raw_data['year'] == 2026].iloc[:len(df_2026)].copy()

            self.df_2026_processed_all = df_2026.copy()

            self.y_2026 = df_2026['target'].values
            self.X_2026 = df_2026.drop(columns=['target', 'year'], errors='ignore')

            if self.model_type == "RF": self._load_rf()
            elif self.model_type == "SVM": self._load_svm()

            self.data_loaded = True
            self.root.after(0, self._on_loading_complete)
            
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Loading Error", str(e))
            self.root.after(0, self.lbl_status.config, {"text": "Status: Error loading data.", "foreground": "#D9534F"})
            self.root.after(0, self.btn_load.config, {"state": "normal"})

    def _load_rf(self):
        config_path = list(self.model_dir.glob("*_config.json"))[0]
        with open(config_path, 'r') as f: self.cfg = json.load(f)
        self.expected_base_features = [f for f in self.cfg['features_used'] if not f.startswith(('KMeans_Dist_', 'PC'))]
        self.X_2026 = self.X_2026.reindex(columns=self.expected_base_features).fillna(0)
        self.scaler = joblib.load(list(self.model_dir.glob("*_scaler.joblib"))[0])
        self.pca = joblib.load(list(self.model_dir.glob("*_pca.joblib"))[0]) if self.cfg.get('pca_applied', False) else None
        self.kmeans = joblib.load(list(self.model_dir.glob("*_kmeans.joblib"))[0]) if self.cfg.get('kmeans_applied', False) else None
        self.model = joblib.load(list(self.model_dir.glob("*_model.joblib"))[0])

    def _load_svm(self):
        self.svm_poly = joblib.load(list(self.model_dir.glob("*_poly.joblib"))[0])
        self.svm_selector = joblib.load(list(self.model_dir.glob("*_selector.joblib"))[0])
        self.svm_scaler_poly = joblib.load(list(self.model_dir.glob("*_scaler_poly.joblib"))[0])
        self.kmeans = joblib.load(list(self.model_dir.glob("*_kmeans.joblib"))[0])
        self.svm_dist_scaler = joblib.load(list(self.model_dir.glob("*_dist_scaler.joblib"))[0])
        
        self.expected_base_features = self.svm_poly.feature_names_in_
        self.X_2026 = self.X_2026.reindex(columns=self.expected_base_features).fillna(0)
        
        config_path = list(self.model_dir.glob("*_config.json"))[0]
        prefix = config_path.name.replace("_config.json", "")
        
        dummy_x = self._transform_svm(self.X_2026.iloc[[0]])
        self.model = PyTorchLinearSVM(dummy_x.shape[1]).to(self.device)
        self.model.load_state_dict(torch.load(self.model_dir / f"{prefix}_model.pth", map_location=self.device, weights_only=True))
        self.model.eval()

    def _on_loading_complete(self):
        self.lbl_status.config(text=f"Status: Ready. Loaded {len(self.X_2026)} matches.", foreground="#00A388")
        print("\n✅ Pipeline Components Loaded Successfully!\n")
        
        match_options = []
        for i in range(len(self.X_2026)):
            row = self.df_2026_raw.iloc[i]
            match_options.append(f"Row {i}: {row.get('winner_name', 'Winner')} vs {row.get('loser_name', 'Loser')} ({row.get('tourney_name', 'Tourney')})")

        self.combo_matches['values'] = match_options
        self.combo_matches.current(0)
        self.combo_matches.config(state="readonly")

        self.btn_predict.config(state="normal")
        self.btn_eval.config(state="normal")
        self.btn_miner.config(state="normal")
        self.btn_load.config(state="normal")
        self.btn_browse.config(state="normal")

    def _transform_rf(self, X_raw):
        X_scaled = self.scaler.transform(X_raw)
        if self.pca is not None: X_scaled = self.pca.transform(X_scaled)
        if self.kmeans is not None:
            X_scaled = np.hstack((X_scaled, self.kmeans.transform(X_scaled)))
        return X_scaled

    def _transform_svm(self, X_raw):
        X_p = self.svm_poly.transform(X_raw)
        X_p = self.svm_selector.transform(X_p)
        X_p = self.svm_scaler_poly.transform(X_p)
        return np.hstack((X_p, self.svm_dist_scaler.transform(self.kmeans.transform(X_p))))

    def _run_inference_on_row(self, idx, single_X):
        if self.model_type == "RF":
            X_proc = self._transform_rf(single_X)
            pred = self.model.predict(X_proc)[0]
            prob = self.model.predict_proba(X_proc)[0][1] * 100 if hasattr(self.model, "predict_proba") else 0.0
        elif self.model_type == "SVM":
            X_proc = self._transform_svm(single_X)
            with torch.no_grad():
                tensor_x = torch.FloatTensor(X_proc).to(self.device)
                p_val = torch.sigmoid(self.model(tensor_x)).cpu().item()
                pred = 1 if p_val >= 0.5 else 0
                prob = p_val * 100
        return pred, prob

    def predict_single_match(self):
        idx = self.combo_matches.current()
        if idx < 0: return
        
        match_info = self.combo_matches.get()
        actual_target = self.y_2026[idx]
        
        pred, win_prob = self._run_inference_on_row(idx, self.X_2026.iloc[[idx]])
        
        print(f"\n▶ Analyzing: {match_info}")
        print(f"  Target Truth: {'WIN (1)' if actual_target == 1 else 'LOSS (0)'}")
        print(f"  Model Predicted: {'WIN (1)' if pred == 1 else 'LOSS (0)'} (Confidence: {win_prob:.1f}%)")
        print(f"  Result: {'✅ CORRECT' if pred == actual_target else '❌ INCORRECT'}")
        print("-" * 55)

    def analyze_upsets(self):
        try:
            max_upsets = int(self.txt_upset_val.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer for max upset matches.")
            return

        print(f"\n⚡ Mining 2026 Table for top {max_upsets} Elo Upsets...")

        elo_cols = [c for c in self.df_2026_processed_all.columns if 'elo' in c.lower()]
        w_elo_key = next((c for c in elo_cols if 'winner' in c or 'w_' in c), None)
        l_elo_key = next((c for c in elo_cols if 'loser' in c or 'l_' in c), None)

        upset_indices = []
        elo_margins = []

        if w_elo_key and l_elo_key:
            for i in range(len(self.df_2026_processed_all)):
                w_elo = self.df_2026_processed_all.iloc[i][w_elo_key]
                l_elo = self.df_2026_processed_all.iloc[i][l_elo_key]
                if w_elo < l_elo:
                    upset_indices.append(i)
                    elo_margins.append(l_elo - w_elo)
        else:
            print("[System Note] No distinct Elo components identified. Cascading to raw ATP rank differences...")
            for i in range(len(self.df_2026_raw)):
                w_rank = pd.to_numeric(self.df_2026_raw.iloc[i].get('winner_rank'), errors='coerce')
                l_rank = pd.to_numeric(self.df_2026_raw.iloc[i].get('loser_rank'), errors='coerce')
                if pd.notna(w_rank) and pd.notna(l_rank) and w_rank > l_rank:
                    upset_indices.append(i)
                    elo_margins.append(w_rank - l_rank)

        if not upset_indices:
            print("❌ No clear upset matches matching criteria found in current 2026 configuration.")
            return

        sorted_arr = np.argsort(elo_margins)[::-1][:max_upsets]
        target_indices = [upset_indices[idx] for idx in sorted_arr]

        print(f" Found {len(target_indices)} upset rows. Sequential match-by-match log:")
        print("=" * 65)

        for match_idx in target_indices:
            raw_row = self.df_2026_raw.iloc[match_idx]
            w_name = raw_row.get('winner_name', 'Winner')
            l_name = raw_row.get('loser_name', 'Loser')
            tourney = raw_row.get('tourney_name', 'Tourney')
            
            actual_target = self.y_2026[match_idx]
            pred, win_prob = self._run_inference_on_row(match_idx, self.X_2026.iloc[[match_idx]])
            
            match_status = "✅ MODEL CAUGHT IT" if pred == actual_target else "❌ MODEL FOOLED BY UPSET"
            
            print(f"▶ Upset Row #{match_idx}: {w_name} def. {l_name}")
            print(f"  Venue: {tourney} | Target Label: {actual_target}")
            print(f"  Model Predicted: {'WIN (1)' if pred == 1 else 'LOSS (0)'} ({win_prob:.1f}% confidence)")
            print(f"  Evaluation: {match_status}")
            print("-" * 65)

    def run_full_evaluation(self):
        print(f"\n--- Running Full Batch Evaluation ({self.model_type}) ---")
        if self.model_type == "RF":
            X_processed = self._transform_rf(self.X_2026)
            y_pred = self.model.predict(X_processed)
            y_prob = self.model.predict_proba(X_processed)[:, 1] if hasattr(self.model, "predict_proba") else None
        elif self.model_type == "SVM":
            X_processed = self._transform_svm(self.X_2026)
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_processed).to(self.device)
                y_prob = torch.sigmoid(self.model(X_tensor)).cpu().numpy().flatten()
                y_pred = (y_prob >= 0.5).astype(int)

              
        # This will now write directly into your UI ScrolledText widget!
        evaluate_model_bias(self.y_2026, y_pred, self.X_2026, y_prob=y_prob)


if __name__ == "__main__":
    root = tk.Tk()
    app = TennisInferenceGUI(root)
    root.mainloop()