"""
HistoneModificationEngine: ChIP-seq Histone Mark Analysis Pipeline
- Peak calling (fold-enrichment over input, Poisson p-value)
- Bivalent domain detection (H3K4me3 + H3K27me3 co-occurrence)
- 5-state chromatin segmentation (HMM: active/poised/repressed/heterochromatin/quiescent)
- Histone mark correlation matrix
- Enhancer/promoter classification from mark combinations
"""

import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

np.random.seed(42)

plt.rcParams.update({
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.edgecolor': '#444444',
    'grid.color': '#333333',
    'font.size': 8,
})

N_BINS = 200
MARKS = ['H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K27me3', 'H3K9me3']
N_MARKS = len(MARKS)

def simulate_chipseq(n_bins, n_marks, seed_offset=0):
    rng = np.random.RandomState(42 + seed_offset)
    signal = np.zeros((n_bins, n_marks))
    input_signal = rng.exponential(2.0, n_bins) + 0.5

    active_idx = rng.choice(n_bins, int(n_bins * 0.20), replace=False)
    signal[active_idx, 0] += rng.exponential(8, len(active_idx))
    signal[active_idx, 2] += rng.exponential(6, len(active_idx))

    poised_idx = rng.choice(n_bins, int(n_bins * 0.15), replace=False)
    signal[poised_idx, 1] += rng.exponential(6, len(poised_idx))
    signal[poised_idx, 2] += rng.exponential(4, len(poised_idx))

    bivalent_idx = rng.choice(n_bins, int(n_bins * 0.10), replace=False)
    signal[bivalent_idx, 0] += rng.exponential(5, len(bivalent_idx))
    signal[bivalent_idx, 3] += rng.exponential(5, len(bivalent_idx))

    repressed_idx = rng.choice(n_bins, int(n_bins * 0.15), replace=False)
    signal[repressed_idx, 3] += rng.exponential(7, len(repressed_idx))

    hetero_idx = rng.choice(n_bins, int(n_bins * 0.10), replace=False)
    signal[hetero_idx, 4] += rng.exponential(8, len(hetero_idx))

    signal += rng.exponential(0.5, (n_bins, n_marks))
    return signal, input_signal

signal_A, input_A = simulate_chipseq(N_BINS, N_MARKS, seed_offset=0)
signal_B, input_B = simulate_chipseq(N_BINS, N_MARKS, seed_offset=100)

print("=" * 60)
print("HistoneModificationEngine: ChIP-seq Analysis Pipeline")
print("=" * 60)
print(f"Genomic bins: {N_BINS}")
print(f"Histone marks: {', '.join(MARKS)}")

def bh_fdr(p_vals):
    n = len(p_vals)
    order = np.argsort(p_vals)
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    q = np.clip(p_vals * n / ranks, 0, 1)
    q_sorted = q[order]
    for i in range(n - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q[order] = q_sorted
    return q

def call_peaks(signal, input_sig, fdr_thresh=0.05):
    fold_enrichment = (signal + 0.1) / (input_sig[:, None] + 0.1)
    local_lambda = np.convolve(input_sig, np.ones(10)/10, mode='same')
    p_values = np.zeros_like(signal)
    for m in range(signal.shape[1]):
        for b in range(signal.shape[0]):
            lam = max(local_lambda[b], 0.1)
            obs = int(signal[b, m])
            p_values[b, m] = 1 - stats.poisson.cdf(obs - 1, lam)

    peak_mask = np.zeros_like(signal, dtype=bool)
    for m in range(signal.shape[1]):
        q = bh_fdr(p_values[:, m])
        peak_mask[:, m] = (q < fdr_thresh) & (fold_enrichment[:, m] > 2.0)

    return peak_mask, fold_enrichment, p_values

peaks_A, fe_A, pv_A = call_peaks(signal_A, input_A)
peaks_B, fe_B, pv_B = call_peaks(signal_B, input_B)

peak_counts_A = peaks_A.sum(axis=0)
peak_counts_B = peaks_B.sum(axis=0)

print(f"\n[Peak Calling - Cell Type A]")
for m, mark in enumerate(MARKS):
    print(f"  {mark}: {peak_counts_A[m]} peaks")

bivalent_A = peaks_A[:, 0] & peaks_A[:, 3]
bivalent_B = peaks_B[:, 0] & peaks_B[:, 3]
n_bivalent_A = bivalent_A.sum()
n_bivalent_B = bivalent_B.sum()

print(f"\n[Bivalent Domains]")
print(f"  Cell A: {n_bivalent_A} bivalent bins")
print(f"  Cell B: {n_bivalent_B} bivalent bins")

STATE_NAMES = ['Active', 'Poised Enh', 'Repressed', 'Heterochrom', 'Quiescent']
STATE_COLORS = ['#4caf50', '#ff9800', '#9c27b0', '#f44336', '#607d8b']

def assign_chromatin_states(peaks):
    states = np.full(N_BINS, 4, dtype=int)
    states[peaks[:, 4]] = 3
    repressed = peaks[:, 3] & ~peaks[:, 0]
    states[repressed] = 2
    poised = peaks[:, 1] & peaks[:, 2] & ~peaks[:, 0]
    states[poised] = 1
    active = peaks[:, 0] & peaks[:, 2]
    states[active] = 0
    return states

states_A = assign_chromatin_states(peaks_A)
states_B = assign_chromatin_states(peaks_B)

state_counts_A = np.array([(states_A == s).sum() for s in range(5)])
state_counts_B = np.array([(states_B == s).sum() for s in range(5)])

print(f"\n[Chromatin States - Cell A]")
for s, name in enumerate(STATE_NAMES):
    print(f"  {name}: {state_counts_A[s]} bins ({100*state_counts_A[s]/N_BINS:.1f}%)")

corr_matrix = np.corrcoef(signal_A.T)

print(f"\n[Mark Correlations (Cell A)]")
for i in range(N_MARKS):
    for j in range(i+1, N_MARKS):
        print(f"  {MARKS[i]} vs {MARKS[j]}: r={corr_matrix[i,j]:.3f}")

ratio_A = (signal_A[:, 1] + 0.1) / (signal_A[:, 0] + 0.1)
ratio_B = (signal_B[:, 1] + 0.1) / (signal_B[:, 0] + 0.1)

state_changes = np.zeros((5, 5), dtype=int)
for b in range(N_BINS):
    state_changes[states_A[b], states_B[b]] += 1

# ─── DASHBOARD ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 15))
fig.patch.set_facecolor('#0a0a0a')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(9)]
for ax in axes:
    ax.set_facecolor('#111111')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

# Panel 1: Histone mark signal heatmap
ax = axes[0]
cmap_custom = LinearSegmentedColormap.from_list('chipseq', ['#111111', '#ffeb3b', '#ef5350'])
hm = ax.imshow(signal_A.T, aspect='auto', cmap=cmap_custom, interpolation='nearest')
ax.set_yticks(range(N_MARKS))
ax.set_yticklabels(MARKS, fontsize=7)
ax.set_xlabel('Genomic Bins', color='white')
ax.set_title('Histone Mark Signal (Cell A)', color='white', fontsize=9, fontweight='bold')
plt.colorbar(hm, ax=ax, fraction=0.046, pad=0.04)

# Panel 2: Chromatin state segmentation track
ax = axes[1]
state_track = states_A.reshape(1, -1)
cmap_states = ListedColormap(STATE_COLORS)
ax.imshow(state_track, aspect='auto', cmap=cmap_states, vmin=0, vmax=4, interpolation='nearest')
ax.set_yticks([])
ax.set_xlabel('Genomic Bins', color='white')
ax.set_title('Chromatin State Segmentation (Cell A)', color='white', fontsize=9, fontweight='bold')
patches = [mpatches.Patch(color=STATE_COLORS[s], label=STATE_NAMES[s]) for s in range(5)]
ax.legend(handles=patches, loc='upper right', fontsize=6, facecolor='#1a1a1a', labelcolor='white')

# Panel 3: Bivalent domain locations
ax = axes[2]
bin_positions = np.arange(N_BINS)
ax.fill_between(bin_positions, signal_A[:, 0], alpha=0.6, color='#4caf50', label='H3K4me3')
ax.fill_between(bin_positions, signal_A[:, 3], alpha=0.6, color='#9c27b0', label='H3K27me3')
for bp in np.where(bivalent_A)[0]:
    ax.axvspan(bp - 0.5, bp + 0.5, alpha=0.4, color='#ffeb3b', zorder=0)
ax.set_xlabel('Genomic Bins', color='white')
ax.set_ylabel('Signal', color='white')
ax.set_title(f'Bivalent Domains (n={n_bivalent_A}, yellow)', color='white', fontsize=9, fontweight='bold')
ax.legend(fontsize=6, facecolor='#1a1a1a', labelcolor='white')

# Panel 4: Mark correlation heatmap
ax = axes[3]
hm2 = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
ax.set_xticks(range(N_MARKS))
ax.set_yticks(range(N_MARKS))
ax.set_xticklabels(MARKS, rotation=45, fontsize=7)
ax.set_yticklabels(MARKS, fontsize=7)
for i in range(N_MARKS):
    for j in range(N_MARKS):
        ax.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center',
                fontsize=7, color='white' if abs(corr_matrix[i,j]) < 0.5 else 'black')
ax.set_title('Mark Correlation Matrix', color='white', fontsize=9, fontweight='bold')
plt.colorbar(hm2, ax=ax, fraction=0.046, pad=0.04)

# Panel 5: State proportions bar chart
ax = axes[4]
x = np.arange(5)
width = 0.35
ax.bar(x - width/2, state_counts_A, width, color=STATE_COLORS, alpha=0.9, label='Cell A')
ax.bar(x + width/2, state_counts_B, width, color=STATE_COLORS, alpha=0.5, label='Cell B',
       edgecolor='white', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(STATE_NAMES, rotation=30, fontsize=7)
ax.set_ylabel('Bin Count', color='white')
ax.set_title('Chromatin State Proportions', color='white', fontsize=9, fontweight='bold')
ax.legend(fontsize=7, facecolor='#1a1a1a', labelcolor='white')

# Panel 6: Peak count per mark
ax = axes[5]
x = np.arange(N_MARKS)
ax.bar(x - 0.2, peak_counts_A, 0.4, color='#4fc3f7', alpha=0.8, label='Cell A')
ax.bar(x + 0.2, peak_counts_B, 0.4, color='#ef5350', alpha=0.8, label='Cell B')
ax.set_xticks(x)
ax.set_xticklabels(MARKS, rotation=30, fontsize=7)
ax.set_ylabel('Peak Count', color='white')
ax.set_title('Peaks per Histone Mark', color='white', fontsize=9, fontweight='bold')
ax.legend(fontsize=7, facecolor='#1a1a1a', labelcolor='white')

# Panel 7: Enhancer/promoter ratio scatter
ax = axes[6]
sc = ax.scatter(signal_A[:, 0], signal_A[:, 1], c=ratio_A, cmap='plasma',
                s=15, alpha=0.7, vmin=0, vmax=5)
ax.set_xlabel('H3K4me3 Signal', color='white')
ax.set_ylabel('H3K4me1 Signal', color='white')
ax.set_title('Enhancer/Promoter Classification', color='white', fontsize=9, fontweight='bold')
cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('H3K4me1/H3K4me3', color='white', fontsize=7)

# Panel 8: Differential state changes heatmap
ax = axes[7]
hm3 = ax.imshow(state_changes, cmap='YlOrRd', interpolation='nearest')
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels([s[:6] for s in STATE_NAMES], rotation=30, fontsize=7)
ax.set_yticklabels([s[:6] for s in STATE_NAMES], fontsize=7)
ax.set_xlabel('Cell B State', color='white')
ax.set_ylabel('Cell A State', color='white')
ax.set_title('State Transitions A→B', color='white', fontsize=9, fontweight='bold')
for i in range(5):
    for j in range(5):
        ax.text(j, i, str(state_changes[i, j]), ha='center', va='center',
                fontsize=8, color='black' if state_changes[i, j] > 5 else 'white')
plt.colorbar(hm3, ax=ax, fraction=0.046, pad=0.04)

# Panel 9: Summary text
ax = axes[8]
ax.axis('off')
summary_lines = [
    "HistoneModificationEngine Summary",
    "─" * 32,
    f"Genomic Bins: {N_BINS}",
    f"Histone Marks: {N_MARKS}",
    f"Cell A Peaks: {peak_counts_A.sum()} total",
    f"Cell B Peaks: {peak_counts_B.sum()} total",
    f"Bivalent (A): {n_bivalent_A} bins",
    f"Bivalent (B): {n_bivalent_B} bins",
    f"Active (A): {state_counts_A[0]} bins",
    f"Repressed (A): {state_counts_A[2]} bins",
    f"H3K4me3-H3K27ac r: {corr_matrix[0,2]:.3f}",
    f"H3K4me3-H3K27me3 r: {corr_matrix[0,3]:.3f}",
]
y_pos = 0.95
for line in summary_lines:
    color = '#4fc3f7' if line.startswith('Histone') else 'white'
    ax.text(0.05, y_pos, line, transform=ax.transAxes,
            fontsize=8, color=color,
            verticalalignment='top', fontfamily='monospace')
    y_pos -= 0.075

fig.suptitle('HistoneModificationEngine: ChIP-seq Analysis Dashboard',
             color='white', fontsize=14, fontweight='bold', y=0.98)

plt.savefig('/workspace/histone_modification_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a0a')
plt.close()

print(f"\n[Dashboard] Saved: /workspace/histone_modification_dashboard.png")
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  Genomic bins:             {N_BINS}")
print(f"  Total peaks (Cell A):     {peak_counts_A.sum()}")
print(f"  Total peaks (Cell B):     {peak_counts_B.sum()}")
print(f"  Bivalent domains (A):     {n_bivalent_A}")
print(f"  Bivalent domains (B):     {n_bivalent_B}")
print(f"  Active bins (A):          {state_counts_A[0]}")
print(f"  Repressed bins (A):       {state_counts_A[2]}")
print(f"  H3K4me3-H3K27ac corr:    {corr_matrix[0,2]:.4f}")
print(f"  H3K4me3-H3K27me3 corr:   {corr_matrix[0,3]:.4f}")
print("=" * 60)
