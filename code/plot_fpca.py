"""Figure 3: FPCA feature visualization"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys, json

sys.path.insert(0, str(Path(__file__).parent))
BASE = Path('/Users/melodysong/code/phd/battery_ml')
OUT  = BASE / 'results/figures'; OUT.mkdir(parents=True, exist_ok=True)

# ── Load features ───────────────────────────────────────────────────────────
li = pd.read_parquet(BASE/'features/li_ion_features.parquet')
zn = pd.read_parquet(BASE/'features/zn_ion_train_features.parquet')

li_fpca = li[['fpca_1','fpca_2','fpca_3','cycle_life']].dropna()
zn_fpca = zn[['fpca_1','fpca_2','fpca_3','cycle_life']].dropna()

# ── Reload FPCA basis to plot components ────────────────────────────────────
V_GRID_LI = np.linspace(2.0, 3.5, 1000)
sev_dir = BASE / 'data/li_ion/severson'
curves = []
for split in ['test1','test2']:
    for f in sorted((sev_dir/split).glob('cell*.csv')):
        arr = np.loadtxt(str(f), delimiter=',')
        if arr.shape[1] >= 5:
            from scipy.signal import savgol_filter
            q5 = arr[:, 4]
            dq = np.gradient(q5, V_GRID_LI)
            dq = savgol_filter(dq, 11, 3)
            curves.append(dq)

X = np.array(curves)
X -= X.mean(axis=1, keepdims=True)

from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA
fd = FDataGrid(X, grid_points=V_GRID_LI)
fpca = FPCA(n_components=3); fpca.fit(fd)
components = fpca.components_.data_matrix[:,:,0]  # (3, 1000)
var = fpca.explained_variance_ratio_

# ── Figure 3 ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
colors = ['#2196F3','#FF9800','#4CAF50']

# Panel A: FPCA basis functions
ax = axes[0,0]
for i, (comp, c) in enumerate(zip(components, colors)):
    ax.plot(V_GRID_LI, comp, color=c, lw=1.8, label=f'PC{i+1} ({var[i]*100:.1f}%)')
ax.axhline(0, color='k', lw=0.5, ls='--')
ax.set_xlabel('Voltage (V)', fontsize=11)
ax.set_ylabel('dQ/dV component (a.u.)', fontsize=11)
ax.set_title('(A) Li-ion FPCA basis functions', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.set_xlim(2.0, 3.5)

# Panel B: Scree plot
ax = axes[0,1]
cumvar = np.cumsum(var)
ax.bar(range(1,4), var*100, color=colors, alpha=0.8, label='Individual')
ax.plot(range(1,4), cumvar*100, 'k-o', lw=2, ms=6, label=f'Cumulative ({cumvar[-1]*100:.1f}%)')
ax.axhline(95, color='gray', ls='--', lw=1, label='95% threshold')
ax.set_xlabel('PC index', fontsize=11); ax.set_ylabel('Variance explained (%)', fontsize=11)
ax.set_title('(B) Scree plot', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.set_xticks([1,2,3])

# Panel C: PC1 vs PC2 scatter
ax = axes[1,0]
sc1 = ax.scatter(li_fpca.fpca_1, li_fpca.fpca_2, c=np.log(li_fpca.cycle_life),
                  cmap='Blues', alpha=0.6, s=30, label=f'Li-ion (n={len(li_fpca)})')
sc2 = ax.scatter(zn_fpca.fpca_1, zn_fpca.fpca_2, c=np.log(zn_fpca.cycle_life),
                  cmap='Oranges', alpha=0.6, s=30, marker='s', label=f'Zn-ion (n={len(zn_fpca)})')
ax.set_xlabel('PC1 score', fontsize=11); ax.set_ylabel('PC2 score', fontsize=11)
ax.set_title('(C) PC1 vs PC2 (color = log cycle life)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
plt.colorbar(sc1, ax=ax, label='log(cycle life) [Li]', fraction=0.03)

# Panel D: Feature-RUL correlations
ax = axes[1,1]
feats = ['fpca_1','fpca_2','fpca_3','delta_Q_var']
feat_labels = ['PC1','PC2','PC3','ΔQ var']
li_corrs = [li[f].corr(np.log(li['cycle_life'])) for f in ['fpca_1','fpca_2','fpca_3'] 
            + (['delta_Q_var'] if 'delta_Q_var' in li.columns else [])]
zn_corrs = [zn[f].corr(np.log(zn['cycle_life'])) for f in feats[:len(li_corrs)]]

x = np.arange(len(li_corrs))
w = 0.35
ax.bar(x - w/2, li_corrs, w, label='Li-ion', color='#2196F3', alpha=0.8)
ax.bar(x + w/2, zn_corrs, w, label='Zn-ion', color='#FF9800', alpha=0.8)
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(feat_labels[:len(li_corrs)], fontsize=10)
ax.set_ylabel('Pearson r with log(cycle life)', fontsize=11)
ax.set_title('(D) Feature–RUL correlations', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(OUT/'fig3_fpca.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUT}/fig3_fpca.png")
