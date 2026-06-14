import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class TeeLogger:
    """
    Duplicates sys.stdout to a log file.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

def plot_benchmark_results(all_results, output_dir):
    """
    Plots benchmark comparisons from the collected results dictionary.
    
    all_results format:
    {
        'nn': {
            'static': {'final_accuracy': 70.0, 'upset_prediction_accuracy': 20.0, ...},
            'retrain': {...},
            ...
        },
        'pcn': { ... }
    }
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Bias and Overall Metrics Comparison
    rows = []
    for model, modes in all_results.items():
        for mode, res in modes.items():
            if 'bias_metrics' in res:
                b = res['bias_metrics']
                rows.append({
                    'Model': model.upper(),
                    'Mode': mode.capitalize(),
                    'Accuracy': b.get('final_accuracy', 0),
                    'Upset Accuracy': b.get('upset_prediction_accuracy', 0),
                    'Elo Reliance': b.get('elo_reliance', 0),
                    'Class 1 Pred Rate': b.get('class_1_prediction_rate', 0)
                })
                
    if rows:
        df = pd.DataFrame(rows)
        # Create a grouped bar chart
        df_melted = df.melt(id_vars=['Model', 'Mode'], 
                            value_vars=['Accuracy', 'Upset Accuracy', 'Elo Reliance', 'Class 1 Pred Rate'],
                            var_name='Metric', value_name='Percentage')
                            
        # Combine Model and Mode for x-axis
        df_melted['Model_Mode'] = df_melted['Model'] + ' ' + df_melted['Mode']
        
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(data=df_melted, x='Metric', y='Percentage', hue='Model_Mode')
        plt.title('Benchmark Bias & Overall Metrics Comparison', fontsize=16, fontweight='bold')
        plt.ylim(0, 105)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.xlabel('')
        plt.legend(title='Model & Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3, fontsize=9)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
        plt.close()
        
    # 2. Player Performance Comparison (Accuracy)
    player_rows = []
    for model, modes in all_results.items():
        for mode, res in modes.items():
            if 'player_metrics' in res:
                for player, p_metrics in res['player_metrics'].items():
                    player_rows.append({
                        'Model_Mode': f"{model.upper()} {mode.capitalize()}",
                        'Player': player,
                        'Accuracy': p_metrics.get('Accuracy', 0),
                        'Matches': p_metrics.get('Matches', 0)
                    })
                    
    if player_rows:
        df_p = pd.DataFrame(player_rows)
        # Sort players alphabetically for consistency
        df_p = df_p.sort_values(by='Player')
        
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(data=df_p, x='Player', y='Accuracy', hue='Model_Mode')
        plt.title('Target Players Prediction Accuracy Comparison', fontsize=16, fontweight='bold')
        plt.ylim(0, 105)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.xlabel('')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Model & Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', padding=3, fontsize=9)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'player_metrics_accuracy.png'), dpi=300)
        plt.close()
