"""
MT-GP transfer learning experiments.
N_target sweep {2,5,10,20,40} x 200 Monte Carlo trials.
Models: MT-GP (ICM), GP-Direct, Ridge, Shuffled-label control.
"""
import sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
import torch
import gpytorch

warnings.filterwarnings('ignore')
BASE = Path('/Users/melodysong/code/phd/battery_ml')
RESULTS = BASE / 'results'; RESULTS.mkdir(exist_ok=True)

# ── Load features ────────────────────────────────────────────────────────────
li = pd.read_parquet(BASE/'features/li_ion_scalar_features.parquet')
zn = pd.read_parquet(BASE/'features/zn_ion_scalar_features.parquet')

# exp_b: early exponential decay rate — transfers across chemistries (r≈-0.4 both)
# delta_Q_var: capacity increment variance — strong Li predictor, weaker for Zn
FEAT_COLS = ['exp_b', 'delta_Q_var']

# Li-ion source: drop NaN, filter obvious outliers
li_src = li.dropna(subset=FEAT_COLS+['cycle_life']).copy()
li_src = li_src[(li_src.cycle_life > 10) & (li_src.exp_b < 4.9)]  # drop boundary hits
X_src = li_src[FEAT_COLS].values.astype(float)
y_src = np.log(li_src['cycle_life'].values.astype(float))
print(f"Li-ion source: {len(li_src)} cells, cycle_life [{li_src.cycle_life.min():.0f},{li_src.cycle_life.max():.0f}]")

# Zn-ion: drop NaN, positive cycle_life
zn_clean = zn.dropna(subset=FEAT_COLS+['cycle_life']).copy()
zn_clean = zn_clean[zn_clean.cycle_life > 0]
X_zn = zn_clean[FEAT_COLS].values.astype(float)
y_zn = np.log(zn_clean['cycle_life'].values.astype(float))
print(f"Zn-ion train pool: {len(zn_clean)} cells, cycle_life [{zn_clean.cycle_life.min():.0f},{zn_clean.cycle_life.max():.0f}]")

# ── GP models ────────────────────────────────────────────────────────────────
class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, X, y, likelihood):
        super().__init__(X, y, likelihood)
        self.mean = gpytorch.means.ConstantMean()
        self.covar = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=X.shape[1]))
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean(x), self.covar(x))

def fit_gp(X_tr, y_tr, n_steps=60, init_state=None):
    """Fit GP, optionally warm-started from a prior state_dict."""
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    lik = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGP(Xt, yt, lik)
    if init_state is not None:
        # Warm-start: load hyperparameters from Li-ion GP (except training data)
        try:
            model.load_state_dict(init_state['model'], strict=False)
            lik.load_state_dict(init_state['lik'],   strict=False)
        except Exception:
            pass
    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    for _ in range(n_steps):
        opt.zero_grad()
        loss = -mll(model(Xt), yt)
        loss.backward(); opt.step()
    return model, lik

def get_state(model, lik):
    """Extract hyperparameters (not training data) for warm-start transfer."""
    return {'model': {k: v.detach().clone()
                      for k,v in model.state_dict().items()
                      if 'train_inputs' not in k and 'train_targets' not in k},
            'lik':   {k: v.detach().clone()
                      for k,v in lik.state_dict().items()}}

def predict_gp(model, lik, X_te):
    model.eval(); lik.eval()
    Xte = torch.tensor(X_te, dtype=torch.float32)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = lik(model(Xte))
    return pred.mean.numpy(), pred.variance.numpy()

# Multi-task GP via concatenated tasks with IndexKernel
class MTGP(gpytorch.models.ExactGP):
    def __init__(self, X, y, likelihood, n_tasks=2):
        super().__init__(X, y, likelihood)
        self.mean = gpytorch.means.ConstantMean()
        data_covar = gpytorch.kernels.MaternKernel(
            nu=2.5, ard_num_dims=X.shape[1]-1)  # -1 because last col is task index
        task_covar = gpytorch.kernels.IndexKernel(num_tasks=n_tasks, rank=1)
        self.covar = gpytorch.kernels.MultitaskKernel(
            data_covar, num_tasks=n_tasks, rank=1) if False else \
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.ProductKernel(data_covar, task_covar))
        # Simpler: use LCMKernel approximation via hadamard
        self.data_covar = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=X.shape[1]-1))
        self.task_covar = gpytorch.kernels.IndexKernel(num_tasks=n_tasks, rank=1)
    def forward(self, x):
        x_feat = x[..., :-1]
        x_task = x[..., -1:].long()
        mean = self.mean(x_feat)
        k_data = self.data_covar(x_feat)
        k_task = self.task_covar(x_task)
        covar = k_data.mul(k_task)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

def fit_mtgp(X_src, y_src_centered, X_tgt, y_tgt_centered, n_steps=200):
    """Fit multi-task GP on pre-centered (zero-mean per task) labels.
    
    Caller must subtract per-task global means before passing y values,
    and add back mu_tgt after prediction. Using global means (not per-trial
    sample means) avoids noisy estimates with small N.
    """
    n_src, n_tgt = len(X_src), len(X_tgt)
    X_all = np.vstack([
        np.hstack([X_src, np.zeros((n_src,1))]),
        np.hstack([X_tgt, np.ones((n_tgt,1))])
    ])
    y_all = np.concatenate([y_src_centered, y_tgt_centered])
    Xt = torch.tensor(X_all, dtype=torch.float32)
    yt = torch.tensor(y_all, dtype=torch.float32)
    lik = gpytorch.likelihoods.GaussianLikelihood()
    model = MTGP(Xt, yt, lik)
    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    for _ in range(n_steps):
        opt.zero_grad()
        try:
            loss = -mll(model(Xt), yt)
            loss.backward(); opt.step()
        except Exception:
            break
    return model, lik

def predict_mtgp(model, lik, X_te, mu_tgt=0.0):
    """Predict on Zn-ion cells (task=1), add back task-specific global mean."""
    model.eval(); lik.eval()
    X_te_task = np.hstack([X_te, np.ones((len(X_te),1))])
    Xte = torch.tensor(X_te_task, dtype=torch.float32)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = lik(model(Xte))
    return pred.mean.numpy() + mu_tgt, pred.variance.numpy()

def rmse_cycles(y_pred_log, y_true_log):
    return float(np.sqrt(mean_squared_error(np.exp(y_true_log), np.exp(y_pred_log))))

def coverage_90(mu_log, var_log, y_true_log):
    std = np.sqrt(var_log)
    lo, hi = mu_log - 1.645*std, mu_log + 1.645*std
    return float(np.mean((y_true_log >= lo) & (y_true_log <= hi)))

# ── Experiment loop ───────────────────────────────────────────────────────────
N_TARGETS = [2, 5, 10, 20, 40]
N_MC = 200
rng = np.random.default_rng(0)

# Per-domain normalization: preserves rank-order correspondence between tasks
# while ensuring both domains occupy the same feature range for the GP kernel.
# Li-ion exp_b spans [0,5], Zn spans [0,1.1] — combined scaler breaks length scales.
scaler_li = StandardScaler().fit(X_src)
scaler_zn = StandardScaler().fit(X_zn)
X_src_s = scaler_li.transform(X_src)   # Li features in Li-standardized space
X_zn_s  = scaler_zn.transform(X_zn)    # Zn features in Zn-standardized space

# Global means for task-specific centering
mu_src_global = float(np.mean(y_src))
mu_zn_global  = float(np.mean(y_zn))
y_src_c = y_src - mu_src_global   # centered Li-ion labels
y_zn_c  = y_zn  - mu_zn_global   # centered Zn-ion labels (for HP-Transfer)

# ── Pre-train Li-ion GP for hyperparameter warm-start ────────────────────────
print("Pre-training Li-ion GP for hyperparameter transfer...")
li_gp, li_lik = fit_gp(X_src_s, y_src_c, n_steps=200)
li_hp_state = get_state(li_gp, li_lik)
print(f"  Li-ion GP trained. Length scale: "
      f"{li_gp.covar.base_kernel.lengthscale.detach().numpy().ravel()}")

records = []
print(f"\nRunning {N_MC} trials × {len(N_TARGETS)} N_targets × 5 models...")

for n_t in N_TARGETS:
    rmse_mt, rmse_gp, rmse_rd, rmse_sh = [], [], [], []
    for trial in range(N_MC):
        idx = rng.choice(len(X_zn_s), size=n_t, replace=False)
        val_mask = np.ones(len(X_zn_s), dtype=bool); val_mask[idx] = False
        X_tr = X_zn_s[idx];  y_tr = y_zn[idx]
        X_val = X_zn_s[val_mask]; y_val = y_zn[val_mask]
        y_val_c = y_zn_c[val_mask]
        if len(X_val) == 0: continue
        y_tr_c = y_zn_c[idx]

        # 1. GP-Direct (random init, Zn only)
        try:
            m, l = fit_gp(X_tr, y_tr_c)
            mu_c, var = predict_gp(m, l, X_val)
            mu = mu_c + mu_zn_global
            rmse_gp.append(rmse_cycles(mu, y_val))
            cov = coverage_90(mu, var, y_val)
        except Exception: rmse_gp.append(np.nan); cov=np.nan
        records.append({'N_target':n_t,'trial':trial,'model':'GP-Direct',
                        'rmse':rmse_gp[-1],'coverage_90':cov})

        # 2. MT-GP (task-specific global-mean-centered labels)
        try:
            m, l = fit_mtgp(X_src_s, y_src_c, X_tr, y_tr_c)
            mu, var = predict_mtgp(m, l, X_val, mu_zn_global)
            rmse_mt.append(rmse_cycles(mu, y_val))
            cov = coverage_90(mu, var, y_val)
        except Exception: rmse_mt.append(np.nan); cov=np.nan
        records.append({'N_target':n_t,'trial':trial,'model':'MT-GP',
                        'rmse':rmse_mt[-1],'coverage_90':cov})

        # 3. HP-Transfer: Zn GP warm-started from Li-ion GP hyperparameters
        try:
            m, l = fit_gp(X_tr, y_tr_c, n_steps=100, init_state=li_hp_state)
            mu_c, var = predict_gp(m, l, X_val)
            mu = mu_c + mu_zn_global
            rmse_hp = rmse_cycles(mu, y_val)
            cov_hp  = coverage_90(mu, var, y_val)
        except Exception: rmse_hp = np.nan; cov_hp = np.nan
        records.append({'N_target':n_t,'trial':trial,'model':'HP-Transfer',
                        'rmse':rmse_hp,'coverage_90':cov_hp})

        # 4. Ridge
        try:
            if n_t >= 3:
                ridge = RidgeCV(alphas=[0.01,0.1,1,10]).fit(X_tr, y_tr_c)
                mu_r = ridge.predict(X_val) + mu_zn_global
            else:
                mu_r = np.full(len(y_val), mu_zn_global)
            rmse_rd.append(rmse_cycles(mu_r, y_val))
        except Exception: rmse_rd.append(np.nan)
        records.append({'N_target':n_t,'trial':trial,'model':'Ridge',
                        'rmse':rmse_rd[-1],'coverage_90':np.nan})

        # 5. Shuffled-label control (for MT-GP)
        try:
            y_src_shuf = rng.permutation(y_src_c)
            m, l = fit_mtgp(X_src_s, y_src_shuf, X_tr, y_tr_c)
            mu, var = predict_mtgp(m, l, X_val, mu_zn_global)
            rmse_sh.append(rmse_cycles(mu, y_val))
        except Exception: rmse_sh.append(np.nan)
        records.append({'N_target':n_t,'trial':trial,'model':'Shuffled',
                        'rmse':rmse_sh[-1],'coverage_90':np.nan})

    hp_vals = [r['rmse'] for r in records if r['N_target']==n_t and r['model']=='HP-Transfer']
    med_mt = np.nanmedian(rmse_mt); med_gp = np.nanmedian(rmse_gp)
    print(f"  N={n_t:2d}: HP-Transfer={np.nanmedian(hp_vals):.1f}  MT-GP={med_mt:.1f}  "
          f"GP-Direct={med_gp:.1f}  Ridge={np.nanmedian(rmse_rd):.1f}  Shuffled={np.nanmedian(rmse_sh):.1f}")

# ── Save results ──────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
df.to_parquet(RESULTS/'experiment_results.parquet', index=False)

# Summary stats + Wilcoxon
from scipy.stats import wilcoxon
summary = {}
for n_t in N_TARGETS:
    summary[str(n_t)] = {}
    for model in ['MT-GP','GP-Direct','Ridge','Shuffled','HP-Transfer']:
        vals = df[(df.N_target==n_t)&(df.model==model)]['rmse'].dropna().values
        summary[str(n_t)][model] = {
            'median': float(np.median(vals)) if len(vals) else None,
            'q1': float(np.percentile(vals,25)) if len(vals) else None,
            'q3': float(np.percentile(vals,75)) if len(vals) else None,
        }
    # Wilcoxon: MT-GP vs GP-Direct
    mt  = df[(df.N_target==n_t)&(df.model=='MT-GP')]['rmse'].dropna().values
    gpd = df[(df.N_target==n_t)&(df.model=='GP-Direct')]['rmse'].dropna().values
    n = min(len(mt),len(gpd))
    if n >= 10:
        try:
            _, p = wilcoxon(mt[:n], gpd[:n])
            summary[str(n_t)]['wilcoxon_mtgp_vs_gpdirect_p'] = float(p)
        except: pass

json.dump(summary, open(RESULTS/'summary_stats.json','w'), indent=2)
print("\n✅ Experiments complete. Results saved.")
print(json.dumps({k:{m:v['median'] for m,v in v2.items() if isinstance(v2[m],dict)}
                  for k,v2 in summary.items()}, indent=2))
