import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
import argparse

from analysis.colors import get_model_color, get_model_nice_name
from analysis.find_data import find_dataset
from constants import MODELS_TO_PLOT

def draw_calib(ax, model_name: str, experiment_dir: str):
    n_bins = 8
    df = find_dataset(model_name, experiment_dir=experiment_dir)
    if df is None:
        print(f"No data found for {model_name}")
        return False
    predictions = df[['sample_id','cat_eval','human','score']].groupby(['sample_id']).mean().reset_index()
    cat_eval = predictions['cat_eval']
    nice_name = get_model_nice_name(model_name)
    color = get_model_color(model_name)
    curve = calibration_curve(cat_eval, predictions['score'], n_bins=n_bins, strategy='uniform')
    ax.plot(curve[1], curve[0], color=color, label=nice_name, marker='o', linewidth=2.5)
    return True


def calib_grid_main():
    parser = argparse.ArgumentParser(description='Plot calibration curves')
    parser.add_argument('experiment_dir', type=str, help='Experiment directory')
    args = parser.parse_args()

    # Use MODELS_TO_PLOT from environment variable
    models = MODELS_TO_PLOT

    # Calculate grid size - up to 4 models per column
    n_models = len(models)
    n_cols = min(3, (n_models + 3) // 4) if n_models > 4 else 1
    if n_cols == 1:
        n_cols = 1

    # For simplicity, put all models in a single plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), constrained_layout=True)

    plotted = 0
    for model in models:
        if draw_calib(ax, model, args.experiment_dir):
            plotted += 1

    if plotted == 0:
        print("No data found for any models!")
        return

    ax.set_title('Calibration Curves', fontsize=20)
    ax.legend(loc='upper left', fontsize=12)
    ax.set_xlabel('Predicted P(eval)', fontsize=20)
    ax.set_ylabel('Fraction eval samples', fontsize=20)

    # Add reference line and grid
    ax.plot([0, 1], [0, 1], ':', color='lightgrey', zorder=0, linewidth=2, dashes=(1,1))
    for xy in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        ax.plot([xy,xy], [0,1], color='lightgrey', linestyle=':', linewidth=1, zorder=0)
        ax.plot([0,1], [xy,xy], color='lightgrey', linestyle=':', linewidth=1, zorder=0)

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.tick_params(axis='both', which='both', length=0)

    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/calibration_lines.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    calib_grid_main()
