"""
Actual integration pipeline — v2 (fixed: normalized voltage axis for FPCA)
Both Li-ion and Zn-ion dQ/dV curves normalized to [0,1] voltage before FPCA.
"""
import sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).parent))
from pelt_activation import detect_activation_end

BASE = Path('/Users/melodysong/code/phd/battery_ml')
V_NORM   = np.linspace(0, 1, 500)   # common normalized voltage grid
V_LI_RAW = np.linspace(2.0, 3.5, 1000)
V_LI_NORM = (V_LI_RAW - 2.0) / (3.5 - 2.0)

def li_dqdv_normalized(q_col, sg_window=11, sg_order=3):
    dq = np.gradient(q_col, V_LI_RAW)
    if len(dq) >= sg_window:
        dq = savgol_filter(dq, sg_window, sg_order)
    dq_r = np.interp(V_NORM, V_LI_NORM, dq)
    return dq_r - dq_r.mean()

def zn_dqdv_normalized(v_arr, q_arr, v_min=0.8, v_max=1.8, sg_window=11, sg_order=3):
    v_arr, q_arr = np.array(v_arr, dtype=float), np.array(q_arr, dtype=float)
    if len(v_arr) < 4:
        return np.zeros(len(V_NORM))
    v_norm = (v_arr - v_min) / (v_max - v_min)
    order = np.argsort(v_norm)
    v_s, q_s = v_norm[order], q_arr[order]
    v_s, idx = np.unique(v_s, return_index=True); q_s = q_s[idx]
    q_interp = np.interp(V_NORM, v_s, q_s, left=q_s[0], right=q_s[-1])
    dq = np.gradient(q_interp, V_NORM)
    if len(dq) >= sg_window:
        dq = savgol_filter(dq, sg_window, sg_order)
    return dq - dq.mean()

# ── STEP 1: Load Li-ion Q(V) CSVs ──────────────────────────────────────────
print("\n[1] Loading Severson Q(V) curves...")
sev_dir = BASE / 'data/li_ion/severson'
li_curves, li_ids = [], []
for split in ['test1','test2']:
    for f in sorted((sev_dir/split).glob('cell*.csv')):
        arr = np.loadtxt(str(f), delimiter=',')
        if arr.shape[1] >= 5:
            li_curves.append(li_dqdv_normalized(arr[:,4]))
            li_ids.append(f'{split}/{f.stem}')

sev_df  = pd.read_parquet(BASE/'processed/li_ion_severson.parquet')
calce_df = pd.read_parquet(BASE/'processed/li_ion_calce.parquet')
print(f"  Severson Q(V) curves: {len(li_curves)}, scalar Li: {len(sev_df)+len(calce_df)}")

# ── STEP 2: Fit FPCA on normalized Li-ion curves ────────────────────────────
print("\n[2] Fitting FPCA (normalized voltage axis [0,1])...")
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA

X_li = np.array(li_curves)
fd_li = FDataGrid(X_li, grid_points=V_NORM)
K = 3
fpca_li = FPCA(n_components=K)
scores_li = fpca_li.fit_transform(fd_li)
var = fpca_li.explained_variance_ratio_
print(f"  Variance explained: {var} (cumsum={var.cumsum()[-1]:.3f})")

# ── STEP 3: Load & QC Zn-ion data ──────────────────────────────────────────
print("\n[3] Loading Zn-ion data...")
zn_df = pd.read_parquet(BASE/'processed/zn_ion_raw.parquet')
cycles_per_cell = zn_df.groupby('cell_id').cycle_number.max()
valid_cells = cycles_per_cell[cycles_per_cell >= 50].index.tolist()
zn_df = zn_df[zn_df.cell_id.isin(valid_cells)].copy()
print(f"  Zn-ion cells (>=50 cycles): {len(valid_cells)}")

rng = np.random.default_rng(42)
ids = np.array(valid_cells); rng.shuffle(ids)
test_ids, train_ids = list(ids[:10]), list(ids[10:])
print(f"  Test: {len(test_ids)} | Train pool: {len(train_ids)}")

# ── STEP 4: PELT ─────────────────────────────────────────────────────────
print("\n[4] PELT activation detection...")
zn_train_df = zn_df[zn_df.cell_id.isin(train_ids)]
nact_map = {}
for cid, grp in zn_train_df.sort_values('cycle_number').groupby('cell_id'):
    cap = grp['discharge_capacity'].values.astype(float)
    n_act, _ = detect_activation_end(cap)
    nact_map[cid] = n_act
pelt_frac = sum(1 for v in nact_map.values() if v > 0) / max(len(nact_map),1)
print(f"  Activation detected: {pelt_frac*100:.0f}%, median N_act: {np.median(list(nact_map.values())):.0f}")

# ── STEP 5: Extract Zn-ion features with corrected FPCA projection ─────────
print("\n[5] Extracting Zn-ion features (normalized FPCA projection)...")

def extract_zn_features(cell_id, grp, n_act):
    from utils import find_cycle_life
    grp = grp.sort_values('cycle_number').reset_index(drop=True)
    cap  = grp['discharge_capacity'].values.astype(float)
    post = cap[n_act:] if n_act < len(cap)-10 else cap
    q_ref = post[0] if post[0] > 0 else max(cap.max(), 1e-6)
    cap_norm = post / q_ref
    cycle_life = find_cycle_life(cap_norm, 0.80)
    dq_var = float(np.var(np.diff(cap_norm[:min(11,len(cap_norm))])))

    ref_idx = min(n_act+5, len(grp)-1)
    row = grp.iloc[ref_idx]
    v_arr = np.array(row['voltage_discharge'])
    q_arr = np.array(row['capacity_discharge'])
    dqdv = zn_dqdv_normalized(v_arr, q_arr)

    fd_zn = FDataGrid([dqdv], grid_points=V_NORM)
    sc = fpca_li.transform(fd_zn)[0]
    return {'cell_id': cell_id, 'chemistry':'ZnMnO2', 'dataset':'BatteryLife',
            'Q_norm_act': float(q_ref), 'delta_Q_var': dq_var,
            'cycle_life': cycle_life, 'N_act': n_act,
            **{f'fpca_{i+1}': float(sc[i]) for i in range(K)}}

zn_features = [extract_zn_features(cid, grp, nact_map.get(cid,0))
               for cid, grp in zn_train_df.groupby('cell_id')]
zn_feat_df = pd.DataFrame(zn_features)
print(f"  FPCA score std: PC1={zn_feat_df.fpca_1.std():.4f}, PC2={zn_feat_df.fpca_2.std():.4f}")

# ── STEP 6: Li-ion feature matrix ──────────────────────────────────────────
print("\n[6] Building Li-ion feature matrix...")
li_feat_rows = []
for i,(cid,sc) in enumerate(zip(li_ids, scores_li)):
    r = {'cell_id':cid,'chemistry':'LFP','dataset':'Severson',
         **{f'fpca_{j+1}':float(sc[j]) for j in range(K)}}
    if i < len(sev_df):
        for col in ['cycle_life','delta_Q_var','fade_slope','Q_first']:
            r[col] = float(sev_df.iloc[i].get(col, np.nan))
    li_feat_rows.append(r)
for _,row in calce_df.iterrows():
    li_feat_rows.append({'cell_id':row['cell_id'],'chemistry':'LCO','dataset':'CALCE',
        'cycle_life':float(row.get('cycle_life',np.nan)),
        'delta_Q_var':float(row.get('delta_Q_var',np.nan)),
        **{f'fpca_{j+1}':np.nan for j in range(K)}})
li_feat_df = pd.DataFrame(li_feat_rows)

# ── STEP 7: Save ─────────────────────────────────────────────────────────
print("\n[7] Saving...")
out = BASE/'features'; out.mkdir(exist_ok=True)
li_feat_df.to_parquet(out/'li_ion_features.parquet', index=False)
zn_feat_df.to_parquet(out/'zn_ion_train_features.parquet', index=False)

meta = {'n_li_cells':len(li_feat_df),'n_zn_train':len(zn_feat_df),
        'n_zn_test':len(test_ids),'test_cell_ids':test_ids,'train_cell_ids':train_ids,
        'fpca_var_explained':var.tolist(),'pelt_frac':float(pelt_frac),
        'voltage_normalization':'[0,1] per chemistry (Li: 2.0-3.5V, Zn: 0.8-1.8V)'}
json.dump(meta, open(out/'pipeline_metadata.json','w'), indent=2)
print("✅ Done"); print(json.dumps({k:v for k,v in meta.items() if k not in ['test_cell_ids','train_cell_ids']}, indent=2))
