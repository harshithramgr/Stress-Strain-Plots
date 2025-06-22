import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline

# Optional: Use grid search to find optimal smoothing factor (lambda)
USE_GRID_SEARCH = True

if USE_GRID_SEARCH:
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error

    def pick_best_lambda(x, y, lam_grid=np.logspace(-9, 0.5, 50), k=5, random_state=0):
        """
        Choose the λ that minimizes k-fold CV MSE.
        Returns the best λ (float).
        """
        x = np.asarray(x)
        y = np.asarray(y)
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        cv_errors = []

        for lam in lam_grid:
            fold_err = []
            for train_idx, test_idx in kf.split(x):
                spline = make_smoothing_spline(x[train_idx], y[train_idx], lam=lam)
                y_pred = spline(x[test_idx])
                fold_err.append(mean_squared_error(y[test_idx], y_pred))
            cv_errors.append(np.mean(fold_err))

        return lam_grid[int(np.argmin(cv_errors))]
else:
    pick_best_lambda = lambda x, y: None  # Use automatic GCV


def smoothen_stress_strain_plot(strain_sorted, stress, smoothing_factor=0.0000000005, save_path=None):
    """
    Plots a smoothed stress–strain curve, finds two critical points,
    appends spline coefficients to 'coefficients.csv',
    and saves the plot to the specified path.
    """
    spline = make_smoothing_spline(strain_sorted, stress, lam=smoothing_factor)
    smoothed_stress = spline(strain_sorted)
    coeffs = spline.c

    # Save coefficients to CSV
    pd.DataFrame([coeffs]).to_csv(
        "coefficients.csv",
        mode="a",
        header=False,
        index=False
    )

    # Derivative (slope) of smoothed curve
    slope = np.diff(smoothed_stress) / np.diff(strain_sorted)

    # Find Point A: positive-to-negative slope change
    point_a_x = point_a_y = None
    for i in range(len(slope) - 1):
        if slope[i] > 0 and slope[i + 1] < 0:
            point_a_x = float(strain_sorted[i + 1])
            point_a_y = float(smoothed_stress[i + 1])
            print(f"Point A is ({point_a_x}, {point_a_y})")
            break

    # Find Point B: negative-to-positive slope change
    point_b_x = point_b_y = None
    if point_a_x is not None:
        for i in range(len(slope) - 1):
            if slope[i] < 0 and slope[i + 1] > 0:
                point_b_x = float(strain_sorted[i + 1])
                point_b_y = float(smoothed_stress[i + 1])
                print(f"Point B is ({point_b_x}, {point_b_y})")
                break
        if point_b_x is None:
            point_b_x = float(strain_sorted[-1])
            point_b_y = float(smoothed_stress[-1])
            print(f"Point B is ({point_b_x}, {point_b_y})")

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(strain_sorted, smoothed_stress, label="Smoothed curve", linewidth=4)
    plt.plot(strain_sorted, stress, label="Raw curve", linewidth=0.5)

    # Highlight and annotate points
    info = []
    if point_a_x is not None:
        plt.scatter(point_a_x, point_a_y, color='red', s=200, zorder=5)
        plt.plot([strain_sorted[0], point_a_x], [smoothed_stress[0], point_a_y],
                 color='green', label="Line predictions", linewidth=4)
        info.append(f"Point A: ({point_a_x:.3f}, {point_a_y:.3f})")

        if point_b_x is not None:
            plt.scatter(point_b_x, point_b_y, color='red', s=200, zorder=5)
            plt.plot([point_a_x, point_b_x], [point_a_y, point_b_y],
                     color='green', linewidth=4)
            plt.hlines(y=point_b_y, xmin=point_b_x, xmax=strain_sorted[-1],
                       color='green', linewidth=4)
            info.append(f"Point B: ({point_b_x:.3f}, {point_b_y:.3f})")

            slope_a = (point_a_y - smoothed_stress[0]) / (point_a_x - strain_sorted[0])
            slope_b = (point_b_y - point_a_y) / (point_b_x - point_a_x)
            info.append(f"Slope A: {slope_a:.3f}")
            info.append(f"Slope B: {slope_b:.3f}")
            info.append(f"Residual: {point_b_y:.3f}")

    # Add analysis info box
    if info:
        plt.text(
            0.02, 0.98,
            "\n".join(info),
            transform=plt.gca().transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8)
        )

    plt.xlabel('Strain')
    plt.ylabel('Stress')
    plt.title('Stress-Strain Curves')
    plt.grid(True)
    plt.legend()

    # Save and close the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()




from pathlib import Path
import os
import pandas as pd

def main():
    file_to_delete = Path("coefficients.csv")  # put your path here
    file_to_delete.unlink(missing_ok=True)

    folder = Path("./curve_pairs")
    files  = [f for f in os.listdir(folder) if f.endswith(".xlsx")]


    sorted_files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))

    error_count   = 0
    error_details = []          # collect (file, message)

    for file in sorted_files:
        try:
            filepath = folder / file
            print(f"\nProcessing: {filepath}")

            df = pd.read_excel(filepath)
            df = (
                df.sort_values(by=df.columns[0])
                  .drop_duplicates(subset=df.columns[0], keep="first")
            )

            strain = df["Strain"].values
            stress = df["Stress"].values

            lam = pick_best_lambda(strain, stress)

            save_path = Path("plots") / f"{file}.png"
            save_path.parent.mkdir(exist_ok=True)

            smoothen_stress_strain_plot(
                strain, stress, smoothing_factor=lam, save_path=save_path
            )

        except Exception as e:
            error_count += 1
            error_details.append((file, str(e)))
            print(f"❌  Skipped {file} — {e}")

    # ---------- summary ----------
    print("\nFinished.")
    print(f"✔️  Success: {len(sorted_files) - error_count}")
    print(f"❌  Errors : {error_count}")

    if error_details:
        print("\nFiles with errors:")
        for fname, msg in error_details:
            print(f"  • {fname}: {msg}")

if __name__ == "__main__":
    main()
