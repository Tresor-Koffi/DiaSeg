"""
Clustering Experiments on Diagonal Segments
============================================
Run K-means, Hierarchical Ward, GMM on GaitDB and GaitNDD segments.
Generate publication-ready figures and LaTeX tables.

Figure strategy
---------------
GaitDB  (primary)  : two-panel — (a) t-SNE scatter | (b) cluster feature
                     profiles.  Panel (b) gives semantic meaning to the
                     blobs in (a) without duplicating the metrics table.
GaitNDD (secondary): single full-width t-SNE scatter only.
                     The table already covers the metrics; a duplicate
                     two-panel layout would be redundant.

Author: Research Team
Date: 2026-03-11
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PUBLICATION STYLE — matches extraction_overview_figure.py exactly
# ============================================================================

plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset':   'stix',
    'font.size':          9,
    'axes.titlesize':     9,
    'axes.labelsize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    8,
    'figure.titlesize':   10,
    'axes.linewidth':     0.8,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'lines.linewidth':    1.2,
    'axes.grid':          True,
    'grid.alpha':         0.25,
    'grid.linewidth':     0.5,
    'grid.linestyle':     '--',
    'legend.framealpha':  0.92,
    'legend.edgecolor':   '0.7',
    'legend.borderpad':   0.4,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.facecolor':  'white',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

# Colourblind-safe cluster palette (also distinguishable in greyscale)
CLUSTER_COLORS = [
    '#1A6FBF',  # blue
    '#2CA02C',  # green
    '#C42B2B',  # red
    '#D6820A',  # amber
    '#7B3FA0',  # purple
]

# ============================================================================
# DTW + SEGMENT EXTRACTION
# ============================================================================

def compute_dtw(seq1, seq2):
    """DTW with backtracked path.  Returns (distance, path, directions)."""
    n, m = len(seq1), len(seq2)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (seq1[i-1] - seq2[j-1]) ** 2
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])

    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i, j))
        _, i, j = min((D[i-1,j-1], i-1, j-1),
                      (D[i-1,j],   i-1, j),
                      (D[i,j-1],   i,   j-1))
    path.append((1, 1))
    path.reverse()

    directions = []
    for k in range(1, len(path)):
        di = path[k][0] - path[k-1][0]
        dj = path[k][1] - path[k-1][1]
        directions.append(
            'diagonal'   if di == 1 and dj == 1 else
            'vertical'   if di == 1               else
            'horizontal'
        )
    return np.sqrt(D[n, m]), path, directions


def extract_segments(path, directions, ell_max=3):
    """Extract maximal diagonal segments with up to ell_max breaks."""
    segments = []
    K = len(path)
    for i in range(1, K - 1):
        # locate strict run boundaries
        if not (directions[i-1] == 'diagonal' and
                (i == 1 or directions[i-2] != 'diagonal')):
            continue
        s, E, breaks = i - 1, {}, 0
        for j in range(i, K):
            if j - 1 < len(directions):
                if directions[j-1] == 'diagonal':
                    E[breaks] = j
                else:
                    breaks += 1
                    if breaks > ell_max:
                        break
        for ell, e in E.items():
            if (i < len(directions) and e - 1 < len(directions) and
                    directions[i-1] == 'diagonal' and
                    directions[e-1] == 'diagonal'):
                L = (e - i + 2) - ell
                if L > ell:
                    segments.append({'L': L, 'ell': ell, 's': s, 'e': e})
    return segments

# ============================================================================
# DATA LOADING
# ============================================================================

def load_subject(filepath):
    """Load stride intervals from a whitespace-delimited file."""
    try:
        data = pd.read_csv(filepath, sep=r'\s+', header=None, engine='python')
        col  = data.iloc[:, 1] if data.shape[1] >= 2 else data.iloc[:, 0]
        vals = pd.to_numeric(col, errors='coerce').dropna().values
        return vals if len(vals) > 50 else None
    except Exception:
        return None

# ============================================================================
# SEGMENT EXTRACTION ACROSS A DATASET
# ============================================================================

def extract_all_segments(data_dir, file_pattern, dataset_name,
                         max_comparisons=15):
    """
    Extract diagonal segments from representative pairwise DTW comparisons.

    Returns
    -------
    list of dict — keys: L, ell, d, t0, p_l
    """
    print(f"\n{'='*60}")
    print(f"Extracting segments — {dataset_name}")
    print(f"{'='*60}")

    files = sorted(Path(data_dir).glob(file_pattern))
    seqs  = [(f, load_subject(f)) for f in files]
    seqs  = [(f, s) for f, s in seqs if s is not None]
    print(f"  Valid sequences : {len(seqs)}")

    if len(seqs) < 2:
        print("  Need at least 2 sequences.")
        return []

    all_segs = []
    n_comp   = 0

    for i in range(min(10, len(seqs))):
        for j in range(i + 1, min(i + 6, len(seqs))):
            if n_comp >= max_comparisons:
                break

            s1 = seqs[i][1]; s1 = (s1 - s1.mean()) / s1.std()
            s2 = seqs[j][1]; s2 = (s2 - s2.mean()) / s2.std()

            dist, path, directions = compute_dtw(s1, s2)
            pl = len(path)

            for seg in extract_segments(path, directions):
                L   = seg['L']
                ell = seg['ell']
                d   = dist / pl * L   # cost variation proxy
                t0  = seg['s']
                all_segs.append(
                    {'L': L, 'ell': ell, 'd': d, 't0': t0, 'p_l': pl}
                )

            n_comp += 1
            if n_comp % 5 == 0:
                print(f"  {n_comp} comparisons — "
                      f"{len(all_segs)} segments so far")

    print(f"  {len(all_segs)} segments from {n_comp} comparisons")
    return all_segs

# ============================================================================
# CLUSTERING
# ============================================================================

def run_clustering(segments, dataset_name):
    """
    Run K-means, Hierarchical Ward, and GMM for k in {2, 3, 4, 5}.

    Returns
    -------
    results_df      : pd.DataFrame  — Silhouette + DB Index per (method, k)
    features_scaled : np.ndarray    — z-scored feature matrix (N × 5)
    """
    print(f"\n{'='*60}")
    print(f"Clustering — {dataset_name}")
    print(f"{'='*60}")

    X = np.array([[s['L'], s['ell'], s['d'], s['t0'], s['p_l']]
                  for s in segments])
    X = StandardScaler().fit_transform(X)
    print(f"  Feature matrix : {X.shape}  (L, ell, d, t0, p_l)")

    rows = []
    configs = [
        ('K-means',
         lambda k: KMeans(n_clusters=k, random_state=42,
                          n_init=10).fit_predict(X)),
        ('Hierarchical',
         lambda k: AgglomerativeClustering(n_clusters=k,
                                           linkage='ward').fit_predict(X)),
        ('GMM',
         lambda k: GaussianMixture(n_components=k, random_state=42,
                                   n_init=10).fit_predict(X)),
    ]

    for method, fit_fn in configs:
        print(f"\n  {method}")
        for k in [2, 3, 4, 5]:
            labels = fit_fn(k)
            sil    = silhouette_score(X, labels)
            db     = davies_bouldin_score(X, labels)
            rows.append({'Method': method, 'k': k,
                         'Silhouette': sil, 'DB_Index': db})
            print(f"    k={k}  Sil={sil:.3f}  DB={db:.3f}")

    return pd.DataFrame(rows), X

# ============================================================================
# FIGURE HELPERS
# ============================================================================

def _panel_label(ax, letter, x=-0.15, y=1.06):
    """Bold IEEE-style panel label in upper-left corner."""
    ax.text(x, y, f'({letter})', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left')


def _scatter_panel(ax, coords, labels, k, pcts, counts,
                   dataset_name, letter='a'):
    """
    t-SNE scatter coloured by cluster.
    Dummy scatter entries keep the legend clean (dot + % only).
    """
    _panel_label(ax, letter)

    for c in range(k):
        mask = labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=CLUSTER_COLORS[c],
                   s=8, alpha=0.55, edgecolors='none',
                   rasterized=True)   # keeps PDF file size small

    for c in range(k):
        ax.scatter([], [], c=CLUSTER_COLORS[c], s=25,
                   label=f'C{c+1}: {pcts[c]:.0f}%  ($n={counts[c]}$)')

    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')
    ax.set_title(f't-SNE projection — {dataset_name} ($k={k}$)')
    ax.legend(loc='lower left',
          bbox_to_anchor=(0.001, 0.2),  # (x, y) in axes coordinates
          markerscale=1.0,
          handletextpad=0.1, labelspacing=0.1,
          framealpha=0.5, edgecolor='0.75')
    ax.set_xticks([])   # t-SNE axes carry no interpretable scale
    ax.set_yticks([])


def _feature_profile_panel(ax, Xs, labels, k, segments, letter='b'):
    """
    Grouped bar chart of normalised mean L, ell, d per cluster (± s.e.).
    Answers "what does each cluster represent?" — information the table
    cannot convey.
    """
    _panel_label(ax, letter)

    feat_names = ['$L$', '$\\ell$', '$d$']
    feat_keys  = ['L', 'ell', 'd']
    raw        = np.array([[s[fk] for fk in feat_keys] for s in segments])

    # normalise each feature to [0, 1] so bars are visually comparable
    feat_min = raw.min(axis=0)
    feat_rng = np.where(raw.max(axis=0) - feat_min > 0,
                        raw.max(axis=0) - feat_min, 1)
    raw_norm = (raw - feat_min) / feat_rng

    n_feats = len(feat_names)
    x       = np.arange(n_feats)
    width   = 0.22
    offsets = np.linspace(-(k - 1) / 2, (k - 1) / 2, k) * width

    for c in range(k):
        mask   = labels == c
        vals   = raw_norm[mask]
        means  = vals.mean(axis=0)
        sems   = vals.std(axis=0) / np.sqrt(mask.sum())
        ax.bar(x + offsets[c], means, width,
               color=CLUSTER_COLORS[c], alpha=0.80,
               edgecolor='white', linewidth=0.5,
               yerr=sems, capsize=2, error_kw={'lw': 0.8},
               label=f'C{c+1}')

    ax.set_xticks(x)
    ax.set_xticklabels(feat_names)
    ax.set_ylabel('Normalised mean ($\\pm$ s.e.)')
    ax.set_title('Cluster feature profiles')
    ax.set_ylim(0, 1.28)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.legend(loc='upper right', handlelength=1.2,
              handletextpad=0.3, labelspacing=0.3)

# ============================================================================
# MAIN FIGURE FUNCTION
# ============================================================================

def make_tsne_figure(features_scaled, segments, dataset_name,
                     k, output_path, show_profiles=True, n_sample=1000):
    """
    Build and save the t-SNE figure.

    Parameters
    ----------
    show_profiles : bool
        True  → two-panel: (a) t-SNE scatter + (b) cluster feature profiles
                IEEE double-column width (7.16 in).
        False → single-panel: t-SNE scatter only.
                IEEE single-column width (3.5 in).
    """
    mode = 'two-panel' if show_profiles else 'single-panel'
    print(f"\n  Building t-SNE figure ({mode}, k={k}) ...")

    # subsample for t-SNE speed
    if len(features_scaled) > n_sample:
        rng        = np.random.default_rng(42)
        idx        = rng.choice(len(features_scaled), n_sample, replace=False)
        Xs         = features_scaled[idx]
        seg_sample = [segments[i] for i in idx]
    else:
        Xs         = features_scaled
        seg_sample = segments

    # cluster on the (sub)sample
    labels = KMeans(n_clusters=k, random_state=42,
                    n_init=10).fit_predict(Xs)
    coords = TSNE(n_components=2, perplexity=30,
                  random_state=42, max_iter=1000).fit_transform(Xs)

    counts = [int(np.sum(labels == c)) for c in range(k)]
    pcts   = [100 * ct / len(labels) for ct in counts]

    if show_profiles:
        # --- two-panel: scatter + feature profiles ---
        fig = plt.figure(figsize=(7.16, 3.4))   # IEEE double-column
        gs  = gridspec.GridSpec(1, 2, figure=fig,
                                wspace=0.42,
                                left=0.09, right=0.97,
                                top=0.90, bottom=0.14)
        _scatter_panel(
            fig.add_subplot(gs[0, 0]),
            coords, labels, k, pcts, counts, dataset_name, letter='a')
        _feature_profile_panel(
            fig.add_subplot(gs[0, 1]),
            Xs, labels, k, seg_sample, letter='b')
    else:
        # --- single-panel: scatter only ---
        fig = plt.figure(figsize=(3.5, 3.2))    # IEEE single-column
        gs  = gridspec.GridSpec(1, 1, figure=fig,
                                left=0.10, right=0.97,
                                top=0.90, bottom=0.14)
        _scatter_panel(
            fig.add_subplot(gs[0, 0]),
            coords, labels, k, pcts, counts, dataset_name, letter='a')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved -> {output_path}")

# ============================================================================
# LaTeX TABLE
# ============================================================================

def save_latex_table(results_df, dataset_name, output_path):
    """
    One booktabs row per method (best k by silhouette).
    Best silhouette and best (lowest) DB Index are bolded.
    """
    best = (results_df
            .sort_values('Silhouette', ascending=False)
            .groupby('Method', sort=False)
            .first()
            .reset_index())

    best_sil = best['Silhouette'].max()
    best_db  = best['DB_Index'].min()

    def fmt(v, best_val):
        s = f'{v:.3f}'
        return f'\\textbf{{{s}}}' if v == best_val else s

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        f'\\caption{{Unsupervised Clustering Performance — {dataset_name}}}',
        f'\\label{{tab:clustering_{dataset_name.lower()}}}',
        r'\small',
        r'\begin{tabular}{lccc}',
        r'\toprule',
        r'\textbf{Method} & \textbf{$k$} & \textbf{Silhouette} & \textbf{DB Index} \\',
        r'\midrule',
    ]

    for method in ['K-means', 'Hierarchical', 'GMM']:
        row = best[best['Method'] == method]
        if row.empty:
            continue
        row = row.iloc[0]
        lines.append(
            f"{method} & {int(row['k'])} & "
            f"{fmt(row['Silhouette'], best_sil)} & "
            f"{fmt(row['DB_Index'], best_db)} \\\\"
        )

    lines += [
        r'\midrule',
        r'\multicolumn{4}{l}{\small Best $k$ per method (highest silhouette).'
        r' \textbf{Bold} = overall best.} \\',
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text('\n'.join(lines), encoding='utf-8')
    print(f"  LaTeX table -> {output_path}")

# ============================================================================
# PIPELINE
# ============================================================================

def run_dataset(data_dir, file_pattern, dataset_name,
                fig_dir, table_dir,
                max_comparisons=15, show_profiles=True):
    """Full pipeline for one dataset: extract → cluster → figure → table."""

    segments = extract_all_segments(
        data_dir, file_pattern, dataset_name, max_comparisons)
    if not segments:
        return

    results_df, X = run_clustering(segments, dataset_name)

    make_tsne_figure(
        X, segments, dataset_name, k=3,
        output_path=f"{fig_dir}/tsne_{dataset_name.lower()}_k3.png",
        show_profiles=show_profiles)

    save_latex_table(
        results_df, dataset_name,
        f"{table_dir}/clustering_table_{dataset_name.lower()}.tex")

    csv_path = Path(table_dir) / f"clustering_results_{dataset_name.lower()}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  CSV -> {csv_path}")


def main():
    FIG_DIR   = "results_clustering/figures"
    TABLE_DIR = "results_clustering"

    # GaitDB — primary dataset
    # Two-panel figure: t-SNE scatter + cluster feature profiles
    run_dataset(
        data_dir      = r"C:\Users\treso\Desktop\SSNTEST\GaitDB",
        file_pattern  = "*.txt",
        dataset_name  = "GaitDB",
        fig_dir       = FIG_DIR,
        table_dir     = TABLE_DIR,
        show_profiles = True,
    )

    # GaitNDD — secondary dataset
    # Single-panel figure: t-SNE scatter only (metrics already in table)
    run_dataset(
        data_dir      = r"C:\Users\treso\Desktop\SSNTEST\GaitNDD",
        file_pattern  = "*.ts",
        dataset_name  = "GaitNDD",
        fig_dir       = FIG_DIR,
        table_dir     = TABLE_DIR,
        show_profiles = False,
    )

    print("\n" + "="*60)
    print("Output files:")
    for f in [
        "results_clustering/figures/tsne_gaitdb_k3.png",
        "results_clustering/figures/tsne_gaitndd_k3.png",
        "results_clustering/clustering_table_gaitdb.tex",
        "results_clustering/clustering_table_gaitndd.tex",
        "results_clustering/clustering_results_gaitdb.csv",
        "results_clustering/clustering_results_gaitndd.csv",
    ]:
        print(f"  {f}")
    print("="*60)


if __name__ == "__main__":
    main()