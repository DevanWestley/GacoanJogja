import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.integrate import quad

def remove_outliers_iqr(df, cols):
    bounds = {}
    mask_outlier = pd.Series(False, index=df.index)

    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        bounds[col] = (lower, upper)
        mask_outlier |= (df[col] < lower) | (df[col] > upper)

    return df[~mask_outlier], df[mask_outlier], bounds

def convert_jam(jam_str):
    h = int(jam_str.split(":")[0])
    if h >= 4:
        return h
    else:
        return h + 24

def manual_pca(X, target_dim=4):
    X = np.array(X, dtype=float)
    X_centered = X - np.mean(X, axis=0)
    N = X_centered.shape[0]
    C = (1/N) * X_centered.T @ X_centered

    eigvals, eigvecs = np.linalg.eig(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    eigvecs_reduced = eigvecs[:, :target_dim]
    X_reduced = X_centered @ eigvecs_reduced

    return X_reduced, C, eigvals, eigvecs

def fourier_fit_plot(x, y, n_harmonics=2):
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    N = len(x_sorted)
    X = np.ones((N, 1))
    for n in range(1, n_harmonics + 1):
        X = np.column_stack([X, np.sin(2 * np.pi * n * x_sorted / max(x_sorted))])
        X = np.column_stack([X, np.cos(2 * np.pi * n * x_sorted / max(x_sorted))])
    coeffs, _, _, _ = np.linalg.lstsq(X, y_sorted, rcond=None)
    y_pred = X @ coeffs

    r2 = r2_score(y_sorted, y_pred)

    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, color='blue', label='Data', alpha=0.7)
    plt.plot(x_sorted, y_pred, color='red', label=f'Fourier fit ({n_harmonics} harmonics)\nR² = {r2:.4f}', linewidth=2)
    plt.xlabel("Jam_encoded")
    plt.ylabel("Waktu antri")
    plt.title("Fourier Fit to Queue Time")
    plt.legend()
    plt.grid(True)
    plt.show()

    return coeffs, r2

def fourier_function_string(coeffs, x_max):
    terms = [f"{coeffs[0]:.4f}"]

    n_harmonics = (len(coeffs) - 1) // 2
    for n in range(1, n_harmonics + 1):
        A = coeffs[2*n - 1]
        B = coeffs[2*n]
        terms.append(f"{A:.4f}*sin({2*n}*pi*x/{x_max})")
        terms.append(f"{B:.4f}*cos({2*n}*pi*x/{x_max})")

    return " + ".join(terms)

def main():
    results = []

    df = pd.read_csv("Gacoan Jogja.csv", na_values=["-"])
    
    df["Hari_encoded"], hari_mapping_arr = pd.factorize(df["Hari"])
    df["Jam_encoded"] = df["Jam"].apply(convert_jam)
    df["Cabang_encoded"], cabang_mapping_arr = pd.factorize(df["Cabang"])
    df["Keterangan_encoded"], ket_mapping_arr = pd.factorize(df["Keterangan"])
    df["Sumber_encoded"], sumber_mapping_arr = pd.factorize(df["Sumber"])

    after = df[["Hari_encoded", "Jam_encoded", "Cabang_encoded",
                "Waktu antri", "Keterangan_encoded", "Sumber_encoded"]]

    cleaned_after, outliers_after, bounds_after = remove_outliers_iqr(after, ["Waktu antri"])

    print("=== Batas Outlier (Waktu antri) ===")
    bounds_table = pd.DataFrame([bounds_after["Waktu antri"]],
                                columns=["Lower Bound", "Upper Bound"])
    print(bounds_table)

    print("=== 10 Data Pertama yang Bersih ===")
    print(cleaned_after.head(10))

    print("=== 10 Data Pertama yang Outlier ===")
    print(outliers_after.head(10))

    X_reduced, C, eigvals, eigvecs = manual_pca(cleaned_after, target_dim=4)

    print("Covariance Matrix C:\n", C)
    print("\nEigenvalues (diagonal of D):\n", eigvals)
    print("\nEigenvectors (columns of V):\n", eigvecs)
    print("\nReduced Data (6D → 4D):\n", X_reduced)

    final = cleaned_after.drop(columns=["Keterangan_encoded", "Sumber_encoded"])
    print("=== Final Data (setelah drop kolom) ===")
    print(final.head(10))

    for cabang in range(7):  
        for hari in range(7):  
            subset = final[(final["Cabang_encoded"] == cabang) & (final["Hari_encoded"] == hari)]

            x = subset["Jam_encoded"].values
            y = subset["Waktu antri"].values

            if len(x) > 0:
                coeffs, r2 = fourier_fit_plot(x, y, n_harmonics=2)
                func_str = fourier_function_string(coeffs, x.max())
            else:
                coeffs, r2, func_str = None, None, None

            results.append({
                "Cabang_encoded": cabang,
                "Hari_encoded": hari,
                "Coefficients": coeffs,
                "R2": r2,
                "Function": func_str
            })

    results_df = pd.DataFrame(results)
    print(results_df)

    results_summary = {}

    for cabang in range(7):
        subset = results_df[results_df['Cabang_encoded'] == cabang]

        if len(subset) == 0 or subset['Coefficients'].isnull().all():
            continue

        coef_array = np.array(subset['Coefficients'].tolist())
        coef_sum = np.sum(coef_array, axis=0)
        coef_mean = coef_sum / len(subset)

        results_summary[cabang] = {
            'sum': coef_sum,
            'mean': coef_mean,
            'count': len(subset)
        }

    for cabang, vals in results_summary.items():
        mean_coef = vals['mean']
        T = 27
        if cabang == 2:
            T = 21
        elif cabang in [4, 5]:
            T = 22

        func_str = f"{mean_coef[0]:.4f} "
        if len(mean_coef) > 1:
            func_str += f"+ ({mean_coef[1]:.4f})*sin(2*pi*x/{T}) + ({mean_coef[2]:.4f})*cos(2*pi*x/{T})"
        if len(mean_coef) > 3:
            func_str += f" + ({mean_coef[3]:.4f})*sin(4*pi*x/{T}) + ({mean_coef[4]:.4f})*cos(4*pi*x/{T})"

        print(f"Cabang {cabang}:")
        print("  Count:", vals['count'])
        print("  Mean coefficients:", mean_coef)
        print("  Function:", func_str)
        print()

    integration_plan = [
        (4, 9, [0,1,3,6]),
        (9, 13, [0,1,2,3,4,5,6]),
        (13, 18, [0,1,2,3,4,5,6]),
        (18, 22, [0,1,2,3,4,5,6]),
        (22, 27, [0,1,3,6])
    ]

    matrix = np.full((7, len(integration_plan)), np.nan)

    for col, (start, end, cabangs) in enumerate(integration_plan):
        for cabang in cabangs:
            mean_coef = results_summary[cabang]['mean']

            T = 27
            if cabang == 2:
                T = 21
            elif cabang in [4,5]:
                T = 22

            def f(x, coef=mean_coef, period=T):
                result = coef[0]
                if len(coef) > 1:
                    result += coef[1]*np.sin(2*np.pi*x/period) + coef[2]*np.cos(2*np.pi*x/period)
                if len(coef) > 3:
                    result += coef[3]*np.sin(4*np.pi*x/period) + coef[4]*np.cos(4*np.pi*x/period)
                return result

            integral_val, _ = quad(f, start, end)
            matrix[cabang, col] = integral_val

    print("Integral matrix (rows=cabang 0-6, columns=ranges):")
    print(matrix)

if __name__ == "__main__":
    main()
