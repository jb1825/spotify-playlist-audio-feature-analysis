"""Cleaning, statistics and graphing of audio features across genres."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import CORE_FEATURES, FEATURES, FIG_DIR, OUT_DIR, RANDOM_STATE

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = "Set2"


# ------------------------------------------------------------------- cleaning
def clean(df: pd.DataFrame, top_n_genres: int = 8, min_tracks: int = 15) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=FEATURES + ["genre"]).drop_duplicates(subset=[c for c in ["track_id"] if c in df])
    df = df[(df["tempo"] > 0)]                                    # tempo 0 = failed detection
    df = df[df["genre"].str.lower() != "unknown"]
    counts = df["genre"].value_counts()
    keep = counts[counts >= min_tracks].head(top_n_genres).index
    return df[df["genre"].isin(keep)].reset_index(drop=True)


# ----------------------------------------------------------------- statistics
def summary_table(df):
    s = df.groupby("genre")[FEATURES].agg(["mean", "std"]).round(3)
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s.insert(0, "n_tracks", df.groupby("genre").size())
    return s.sort_values("energy_mean", ascending=False)


def anova_tests(df):
    """One-way ANOVA + eta-squared effect size for each feature across genres."""
    rows = []
    for f in FEATURES:
        groups = [g[f].values for _, g in df.groupby("genre")]
        F, p = stats.f_oneway(*groups)
        grand = df[f].mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_total = ((df[f] - grand) ** 2).sum()
        rows.append(dict(feature=f, F=round(F, 1), p_value=float(p), eta_squared=round(ss_between / ss_total, 3)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- plots
def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_boxplots(df):
    order = df.groupby("genre")["energy"].median().sort_values(ascending=False).index
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, f in zip(axes, CORE_FEATURES):
        sns.boxplot(data=df, x="genre", y=f, order=order, hue="genre", hue_order=order, palette=PALETTE, legend=False, ax=ax, fliersize=2)
        ax.set_title(f"{f.title()} by genre")
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("")
    return _save(fig, "01_boxplots_core_features.png")


def plot_heatmap(df):
    means = df.groupby("genre")[FEATURES].mean()
    scaled = (means - means.min()) / (means.max() - means.min())     # min-max per feature for comparability
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.heatmap(scaled, annot=means.round(2), fmt="", cmap="viridis", cbar_kws={"label": "relative level (min-max scaled)"}, ax=ax)
    ax.set_title("Mean audio features by genre (cell text = actual mean)")
    ax.set_ylabel("")
    return _save(fig, "02_genre_feature_heatmap.png")


def plot_scatter(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="danceability", y="energy", hue="genre", palette=PALETTE, alpha=0.5, s=18, ax=ax)
    ax.set_title("Danceability vs Energy")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="genre")
    return _save(fig, "03_danceability_vs_energy.png")


def plot_tempo_distribution(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.violinplot(data=df, x="genre", y="tempo", hue="genre", palette=PALETTE, legend=False, inner="quartile", cut=0, ax=ax)
    ax.set_title("Tempo (BPM) distribution by genre")
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("")
    return _save(fig, "04_tempo_distribution.png")


def plot_correlation(df):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(df[FEATURES].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation between audio features")
    return _save(fig, "05_correlation_matrix.png")


def plot_radar(df):
    cols = ["danceability", "energy", "valence", "acousticness"]
    means = df.groupby("genre")[cols].mean()
    angles = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = sns.color_palette(PALETTE, len(means))
    for color, (genre, row) in zip(colors, means.iterrows()):
        vals = row.tolist() + row.tolist()[:1]
        ax.plot(angles, vals, label=genre, color=color, linewidth=1.8)
        ax.fill(angles, vals, color=color, alpha=0.06)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.title() for c in cols])
    ax.set_ylim(0, 1)
    ax.set_title("Genre audio fingerprints", pad=20)
    ax.legend(bbox_to_anchor=(1.15, 1.05))
    return _save(fig, "06_radar_genre_profiles.png")


def plot_clusters(df):
    """Unsupervised check: do tracks cluster by genre using audio features alone?"""
    X = StandardScaler().fit_transform(df[FEATURES])
    k = df["genre"].nunique()
    labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    purity = (pd.crosstab(labels, df["genre"]).max(axis=1).sum()) / len(df)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=df["genre"], palette=PALETTE, s=14, alpha=0.6, ax=axes[0])
    axes[0].set_title("PCA of audio features (coloured by true genre)")
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels, palette="tab10", s=14, alpha=0.6, ax=axes[1], legend=False)
    axes[1].set_title(f"K-Means clusters (k={k}), purity = {purity:.2f}")
    for ax in axes:
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    _save(fig, "07_pca_kmeans_clusters.png")
    return round(float(purity), 3)


# ------------------------------------------------------------------- pipeline
def run_all(df: pd.DataFrame, label: str = "") -> dict:
    OUT_DIR.mkdir(exist_ok=True)
    df = clean(df)
    summ = summary_table(df)
    anova = anova_tests(df)
    summ.to_csv(OUT_DIR / "genre_summary_stats.csv")
    anova.to_csv(OUT_DIR / "anova_results.csv", index=False)
    df.to_csv(OUT_DIR / "cleaned_dataset.csv", index=False)

    plot_boxplots(df); plot_heatmap(df); plot_scatter(df); plot_tempo_distribution(df)
    plot_correlation(df); plot_radar(df)
    purity = plot_clusters(df)

    corr = df[FEATURES].corr()
    results = dict(
        source=label, n_tracks=int(len(df)), n_genres=int(df["genre"].nunique()),
        genres=list(summ.index),
        corr_energy_loudness=round(float(corr.loc["energy", "loudness"]), 3),
        corr_energy_acousticness=round(float(corr.loc["energy", "acousticness"]), 3),
        corr_danceability_valence=round(float(corr.loc["danceability", "valence"]), 3),
        kmeans_purity=purity,
        most_danceable=summ["danceability_mean"].idxmax(), least_danceable=summ["danceability_mean"].idxmin(),
        highest_energy=summ["energy_mean"].idxmax(), lowest_energy=summ["energy_mean"].idxmin(),
        fastest=summ["tempo_mean"].idxmax(), slowest=summ["tempo_mean"].idxmin(),
    )
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    return results
