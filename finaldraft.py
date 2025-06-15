import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline

# ------------------------------------------------------------------
# OPTIONAL: hyper‑parameter search for smoothing factor (λ)
# ------------------------------------------------------------------
# You may turn this on/off with the USE_GRID_SEARCH flag.
USE_GRID_SEARCH = True

if USE_GRID_SEARCH:
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error

    def pick_best_lambda(x, y,
                         lam_grid=np.logspace(-9, 0.5, 50),
                         k=5, random_state=0):
        """
        Choose the λ that minimises k‑fold CV MSE.
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

        best_lam = lam_grid[int(np.argmin(cv_errors))]
        return best_lam
else:
    # If grid search is off, None tells SciPy to use GCV automatically.
    pick_best_lambda = lambda x, y: None


# ------------------------------------------------------------------
#  ORIGINAL FUNCTION (unchanged except docstring comment)
# ------------------------------------------------------------------
def smoothen_stress_strain_plot(strain_sorted, stress, smoothing_factor=0.0000000005):
    """
    Plots a smoothed stress–strain curve, finds two critical points,
    and prints slopes/residuals.
    (Body is your original code.)
    """

    first_point_found = False

    # Apply smoothing spline
    spline = make_smoothing_spline(strain_sorted, stress, lam=smoothing_factor)
    smoothed_stress_spline = spline(strain_sorted)

    # Convert to numpy array for calculations
    smoothed_stress = np.array(smoothed_stress_spline)

    # Calculate slope (derivative) at each point
    slope = np.diff(smoothed_stress) / np.diff(strain_sorted)

    # Find first critical point (positive to negative slope transition)
    for i in range(len(slope) - 1):
        if slope[i] > 0 and slope[i + 1] < 0:
            point_a_x = float(strain_sorted[i + 1])
            point_a_y = float(smoothed_stress[i + 1])
            print(f"Point A is ({point_a_x}, {point_a_y})")
            first_point_found = True
            break

    # Find second critical point (negative to positive slope transition)
    point_b_x = None
    point_b_y = None
    if first_point_found:
        for i in range(len(slope) - 1):
            print(slope[i])
            if (slope[i] < 0 and slope[i + 1] > 0):
                point_b_x = float(strain_sorted[i + 1])
                point_b_y = float(smoothed_stress[i + 1])
                print(f"Point B is ({point_b_x}, {point_b_y})")
                break

        if point_b_x is None:
            print("Point B is None so override")
            point_b_x = float(strain_sorted[-1])
            point_b_y = float(smoothed_stress[-1])
            print(f"Point B is ({point_b_x}, {point_b_y})")

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot smoothed curve
    plt.plot(strain_sorted, smoothed_stress_spline,
             label="Smoothed curve", linewidth=4)

    # Plot raw data
    plt.plot(strain_sorted, stress,
             label="Raw curve", linewidth=0.5)

    # Highlight critical points
    if first_point_found:
        plt.scatter(point_a_x, point_a_y, color='red', s=200, zorder=5)

        # Draw analysis lines
        plt.plot([strain_sorted[0], point_a_x], [smoothed_stress[0], point_a_y],
                 color='green', label="Line predictions", linewidth=4)

        if point_b_x is not None and point_b_y is not None:
            plt.scatter(point_b_x, point_b_y, color='red', s=200, zorder=5)
            plt.plot([point_a_x, point_b_x], [point_a_y, point_b_y],
                     color='green', linewidth=4)
            plt.hlines(y=point_b_y, xmin=point_b_x, xmax=strain_sorted[-1],
                       color='green', linewidth=4)

    # Configure plot appearance
    plt.xlabel('Strain')
    plt.ylabel('Stress')
    plt.title('Stress-Strain Curves')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Calculate and display analysis results
    if first_point_found:
        slope_line_a = (point_a_y - smoothed_stress[0]) / (point_a_x - strain_sorted[0])
        print(f"Slope of the first line is {slope_line_a}")

        if point_b_x is not None and point_b_y is not None:
            slope_line_b = (point_b_y - point_a_y) / (point_b_x - point_a_x)
            print(f"Slope of the second line is {slope_line_b}")
            print(f"The residual is {point_b_y}")


# ------------------------------------------------------------------
#  ORIGINAL main() — only lines added are the λ selection + print
# ------------------------------------------------------------------
def main():
    # Load data from CSV file
    values = pd.read_csv('askdatasample6.csv')

    # Check for duplicates
    values_cleaned = values.sort_values(by=values.columns[0]) \
                           .drop_duplicates(subset=values.columns[0], keep='first')

    # Extract strain and stress columns
    strain = values_cleaned['Strain'].values
    stress = values_cleaned['Stress'].values

    # --- NEW: choose smoothing factor automatically ---
    best_lambda = pick_best_lambda(strain, stress)
    if best_lambda is None:
        print("SciPy will choose λ via GCV …")
    else:
        print(f"Chosen λ (cross‑validated): {best_lambda:.3e}")

    # Perform analysis with the selected λ
    smoothen_stress_strain_plot(strain, stress, smoothing_factor=best_lambda)


if __name__ == "__main__":
    main()