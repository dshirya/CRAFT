import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cross_decomposition import PLSRegression

def plot_confidence_ellipse(x, y, ax, n_std=2.0, edgecolor='black', facecolor='none', alpha=1.0, **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must have the same size")
    
    cov = np.cov(x, y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    eigvals, eigvecs = np.linalg.eig(cov)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    
    ellipse = Ellipse((mean_x, mean_y), width=width, height=height,
                      angle=angle, edgecolor=edgecolor,
                      facecolor=facecolor, lw=2, alpha=alpha, **kwargs)
    ax.add_patch(ellipse)
    return ellipse

def run_pls_da(df, output_loadings_excel="outputs/PLS_DA_loadings.xlsx"):
    # -----------------------------
    # Prepare the data for PLS-DA
    # -----------------------------
    # Drop non-numerical columns: "Formula", "Site", "Site_Label"
    X = df.drop(columns=["Formula", "Site", "Site_Label"])
    y = df["Site_Label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    minmax_scaler = MinMaxScaler()
    X_normalized = minmax_scaler.fit_transform(X_scaled)
    
    y_dummies = pd.get_dummies(y)
    Y = y_dummies.values  # shape: (n_samples, n_classes)
    
    # -------------------------------------
    # Fit the PLS-DA model with 2 components
    # -------------------------------------
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_normalized, Y)
    
    X_scores = pls.x_scores_
    
    # ---------------------------------------
    # Compute explained variance (for labels)
    # ---------------------------------------
    total_variance = np.sum(np.var(X_normalized, axis=0))
    explained_variances = np.var(X_scores, axis=0)
    explained_ratio = explained_variances / total_variance * 100  # array of length 2
    
    # -----------------
    # Generate scatter plot with ellipsoids
    # -----------------
    colors = [
        "#c3121e",  # Sangre
        "#0348a1",  # Neptune
        "#ffb01c",  # Pumpkin
        "#027608",  # Clover
        "#1dace6",  # Cerulean
        "#9c5300",  # Cocoa
        "#9966cc",  # Amethyst
    ]
    
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=500)
    
    unique_classes = y.unique()
    color_map = {cls: colors[i % len(colors)] for i, cls in enumerate(unique_classes)}
    
    for cls in unique_classes:
        idx = (y == cls)
        points = X_scores[idx, :]
        ax.scatter(points[:, 0], points[:, 1],
                   color=color_map[cls], label=cls, s=250, alpha=0.5, edgecolors="none")
        plot_confidence_ellipse(points[:, 0], points[:, 1], ax, n_std=2.0,
                        edgecolor=color_map[cls],
                        facecolor=color_map[cls],
                        alpha=0.2)
    
    ax.set_xlabel(f"LV1 ({explained_ratio[0]:.1f}%)", fontsize=18)
    ax.set_ylabel(f"LV2 ({explained_ratio[1]:.1f}%)", fontsize=18)
    
    legend = ax.legend(fontsize=18)
    for text in legend.get_texts():
        text.set_fontstyle('italic')
    
    plt.tick_params(axis='both', labelsize=18)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/PLS_DA_Scatter_Plot.svg", dpi=500)
    plt.show()
    
    # --------------------------------------
    # Compute and output full loadings per component
    # --------------------------------------
    x_loadings = pls.x_loadings_  # shape: (n_features, 2)
    loadings_df = pd.DataFrame(
        x_loadings,
        columns=[f"Component_{i+1}_Loading" for i in range(pls.n_components)]
    )
    loadings_df.insert(0, "Feature", X.columns.tolist())
    
    os.makedirs(os.path.dirname(output_loadings_excel), exist_ok=True)
    loadings_df.to_excel(output_loadings_excel, index=False)
    print(f"Full loadings saved to {output_loadings_excel}.")
    
    # ------------------------------------------------
    # Create top_contribution.xlsx listing only top 10
    # ------------------------------------------------
    # For each component, sort features by absolute loading (descending)
    comp1_loadings = loadings_df["Component_1_Loading"].abs()
    comp2_loadings = loadings_df["Component_2_Loading"].abs()
    
    sorted_feats_comp1 = loadings_df.loc[
        comp1_loadings.sort_values(ascending=False).index, "Feature"
    ].tolist()[:10]  # only top 10
    sorted_feats_comp2 = loadings_df.loc[
        comp2_loadings.sort_values(ascending=False).index, "Feature"
    ].tolist()[:10]  # only top 10
    
    # Build a DataFrame whose columns are the top 10 features for each component
    header1 = f"Component 1 ({explained_ratio[0]:.1f}%)"
    header2 = f"Component 2 ({explained_ratio[1]:.1f}%)"
    
    top_contrib_df = pd.DataFrame({
        header1: pd.Series(sorted_feats_comp1),
        header2: pd.Series(sorted_feats_comp2)
    })
    
    top_contrib_path = os.path.join("outputs", "top_contribution.xlsx")
    top_contrib_df.to_excel(top_contrib_path, index=False)
    print(f"Top 10 features saved to {top_contrib_path}.")
    
    return loadings_df