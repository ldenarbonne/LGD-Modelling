"""
Core LGD modelling functions shared by the Streamlit app.

They reproduce the methodology of notebooks/lgd_modelling.ipynb:
sample selection, Beta fits, boundary-corrected kernel density estimators,
expert-judgement scorecard and two-stage (logit + Beta regression) benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, stats
from scipy.special import logit
import statsmodels.api as sm
from statsmodels.othermod.betareg import BetaModel
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "lgd_lendingclub_2007_2015.csv.gz"
SEED = 42

# Non-uniform grid on (0,1), denser near the bounds
GRID = np.unique(np.r_[np.geomspace(1e-6, 0.02, 300), np.linspace(0.02, 0.98, 800), 1 - np.geomspace(1e-6, 0.02, 300)])


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_sample(path: Path = DATA_PATH, years: tuple[int, int] = (2008, 2015)) -> pd.DataFrame:
    """Load the prepared sample and keep defaults whose workout can be considered complete."""
    raw = pd.read_csv(path)
    raw["default_year"] = raw["default_year"].astype("Int64")
    df = raw[raw["default_year"].between(*years)].copy()
    df["log_inc"] = np.log1p(df["annual_inc"])
    df["log_ead"] = np.log(df["ead"])
    df["owner"] = df["home_ownership"].isin(["OWN", "MORTGAGE"]).astype(int)
    df["small_bus"] = (df["purpose"] == "small_business").astype(int)
    df["term60"] = (df["term_months"] == 60).astype(int)
    df = df.dropna(subset=["annual_inc", "dti", "revol_util", "int_rate"])
    df["default_year"] = df["default_year"].astype(int)
    return df


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same 70/30 split, stratified by default year, as in the notebook."""
    train, test = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df["default_year"])
    return train.copy(), test.copy()


def continuous_part(rr: np.ndarray) -> np.ndarray:
    return rr[(rr > 0) & (rr < 1)]


# --------------------------------------------------------------------------------------
# Parametric: Beta distribution
# --------------------------------------------------------------------------------------
def beta_mom(y: np.ndarray) -> tuple[float, float]:
    m, v = y.mean(), y.var(ddof=1)
    c = m * (1 - m) / v - 1
    return m * c, (1 - m) * c


def beta_mle(y: np.ndarray) -> tuple[float, float]:
    a, b, _, _ = stats.beta.fit(y, floc=0, fscale=1)
    return a, b


# --------------------------------------------------------------------------------------
# Non-parametric: kernel density estimators on [0,1]
# --------------------------------------------------------------------------------------
def kde_gauss(x, data, h, chunk=400):
    out = np.empty(len(x))
    for i in range(0, len(x), chunk):
        out[i:i + chunk] = stats.norm.pdf((x[i:i + chunk, None] - data[None, :]) / h).mean(axis=1) / h
    return out


def kde_reflect(x, data, h):
    return kde_gauss(x, data, h) + kde_gauss(x, -data, h) + kde_gauss(x, 2 - data, h)


def kde_logit(x, data, h):
    return kde_gauss(logit(x), logit(data), h) / (x * (1 - x))


def kde_beta(x, data, b, chunk=200):
    out = np.empty(len(x))
    for i in range(0, len(x), chunk):
        xx = x[i:i + chunk, None]
        out[i:i + chunk] = stats.beta.pdf(data[None, :], xx / b + 1, (1 - xx) / b + 1).mean(axis=1)
    return out


KERNELS = {
    "Naive Gaussian kernel":        (kde_gauss,   np.geomspace(1e-4, 0.03, 12)),
    "Gaussian kernel + reflection": (kde_reflect, np.geomspace(1e-4, 0.03, 12)),
    "Gaussian kernel on logit":     (kde_logit,   np.geomspace(2e-3, 0.5, 12)),
    "Beta kernel (Chen)":           (kde_beta,    np.geomspace(1e-6, 2e-3, 12)),
}


# Bandwidths selected by likelihood cross-validation in the notebook (8,000 / 4,000 points);
# used as defaults so the app does not have to re-run the cross-validation at start-up.
DEFAULT_BW = {
    "Naive Gaussian kernel": 7.67e-4,
    "Gaussian kernel + reflection": 7.67e-4,
    "Gaussian kernel on logit": 2.13e-2,
    "Beta kernel (Chen)": 8.77e-6,
}


def likelihood_cv(y: np.ndarray, n_fit=4000, n_val=2000, seed=SEED) -> dict[str, dict]:
    """Likelihood cross-validation of each kernel's bandwidth on two disjoint subsamples."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(y)
    fit_s, val_s = perm[:n_fit], perm[n_fit:n_fit + n_val]
    out = {}
    for name, (fn, hs) in KERNELS.items():
        ll = np.array([np.log(np.maximum(fn(val_s, fit_s, h), 1e-12)).mean() for h in hs])
        out[name] = {"grid": hs, "ll": ll, "best": float(hs[ll.argmax()])}
    return out


def normalise(f: np.ndarray) -> np.ndarray:
    return f / integrate.trapezoid(f, GRID)


def cdf_from_density(f: np.ndarray) -> np.ndarray:
    F = integrate.cumulative_trapezoid(f, GRID, initial=0)
    return F / F[-1]


def mixed_rr_quantile(q: float, F_cont: np.ndarray, p0: float, p1: float) -> float:
    """Quantile q of RR under p0·δ0 + (1-p0-p1)·F_cont + p1·δ1."""
    if q <= p0:
        return 0.0
    if q >= 1 - p1:
        return 1.0
    return float(np.interp((q - p0) / (1 - p0 - p1), F_cont, GRID))


def density_metrics(f: np.ndarray, y_test: np.ndarray, rr_test: np.ndarray, p0: float, p1: float) -> dict:
    f = normalise(f)
    F = cdf_from_density(f)
    emp_cdf = np.searchsorted(np.sort(y_test), GRID, side="right") / len(y_test)
    q = lambda p: 1 - mixed_rr_quantile(p, F, p0, p1)
    return {
        "mean log-lik. (test)": float(np.log(np.maximum(np.interp(y_test, GRID, f), 1e-12)).mean()),
        "KS (test)": float(np.abs(F - emp_cdf).max()),
        "LGD q25": q(0.75), "LGD median": q(0.50), "LGD q75": q(0.25),
    }


# --------------------------------------------------------------------------------------
# Expert-judgement scorecard
# --------------------------------------------------------------------------------------
@dataclass
class ScorecardConfig:
    """Points per bucket (more points = higher expected recovery = lower LGD)."""
    repaid: list = field(default_factory=lambda: [0, 10, 20, 30])     # <15%, 15–35%, 35–55%, ≥55%
    mob: list = field(default_factory=lambda: [0, 6, 12])             # ≤9, 10–18, >18 months
    term36: int = 8                                                   # 36-month loan (60-month = 0)
    owner: int = 8                                                    # owner/mortgage (renter/other = 0)
    not_small_bus: int = 6                                            # purpose other than small business
    income: list = field(default_factory=lambda: [0, 5, 10])          # <40k, 40–75k, ≥75k
    n_pools: int = 5

    @property
    def max_score(self) -> int:
        return max(self.repaid) + max(self.mob) + self.term36 + self.owner + self.not_small_bus + max(self.income)


def expert_score(d: pd.DataFrame, cfg: ScorecardConfig) -> pd.Series:
    s = pd.Series(0.0, index=d.index)
    s += np.select([d.pct_principal_repaid < .15, d.pct_principal_repaid < .35, d.pct_principal_repaid < .55],
                   cfg.repaid[:3], cfg.repaid[3])
    s += np.select([d.mob_at_default <= 9, d.mob_at_default <= 18], cfg.mob[:2], cfg.mob[2])
    s += np.where(d.term_months == 60, 0, cfg.term36)
    s += np.where(d.owner == 1, cfg.owner, 0)
    s += np.where(d.small_bus == 1, 0, cfg.not_small_bus)
    s += np.select([d.annual_inc < 40_000, d.annual_inc < 75_000], cfg.income[:2], cfg.income[2])
    return s


def pool_edges(train_score: pd.Series, n_pools: int) -> np.ndarray:
    qs = np.linspace(0, 1, n_pools + 1)[1:-1]
    cuts = np.unique(np.quantile(train_score, qs))
    return np.r_[-np.inf, cuts, np.inf]


def assign_pools(score: pd.Series, edges: np.ndarray) -> pd.Series:
    k = len(edges) - 1
    labs = [f"G{k - i}" for i in range(k)]                 # highest score -> G1 (lowest LGD)
    cat = pd.CategoricalDtype([f"G{i}" for i in range(1, k + 1)], ordered=True)
    return pd.cut(score, edges, labels=labs, right=False).astype(str).astype(cat)


def calibrate_pools(train: pd.DataFrame, B: int = 300, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def boot_q90(s):
        v = s.to_numpy()
        return np.quantile([rng.choice(v, len(v)).mean() for _ in range(B)], 0.90)

    g = train.groupby("pool", observed=True)
    calib = g.agg(score_min=("score", "min"), score_max=("score", "max"), n=("lgd", "size"), lgd_lra=("lgd", "mean"))
    calib["MoC"] = g["lgd"].apply(boot_q90) - calib["lgd_lra"]
    calib["lgd_estimate"] = (calib["lgd_lra"] + calib["MoC"]).clip(upper=1)
    return calib.sort_index()


# --------------------------------------------------------------------------------------
# Two-stage parametric benchmark: logit P(RR=0) + Beta regression on RR>0
# --------------------------------------------------------------------------------------
FEATS = ["pct_principal_repaid", "mob_at_default", "term60", "owner", "small_bus", "log_inc", "int_rate", "dti", "log_ead"]


def design(d: pd.DataFrame) -> pd.DataFrame:
    X = d[FEATS].astype(float).copy()
    X["mob_at_default"] = X["mob_at_default"].clip(upper=60)
    return sm.add_constant(X, has_constant="add")


@dataclass
class TwoStageModel:
    logit_params: np.ndarray
    beta_params: np.ndarray
    columns: list

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        X = design(d)[self.columns].to_numpy()
        p0 = 1 / (1 + np.exp(-X @ self.logit_params))
        mu = 1 / (1 + np.exp(-X @ self.beta_params[:X.shape[1]]))
        return p0 + (1 - p0) * (1 - mu)


def fit_two_stage(train: pd.DataFrame) -> tuple[TwoStageModel, pd.DataFrame]:
    X = design(train)
    lm = sm.Logit((train.rr == 0).astype(int), X).fit(disp=0)
    pos = train.rr > 0
    n_pos = pos.sum()
    y_pos = (train.loc[pos, "rr"] * (n_pos - 1) + 0.5) / n_pos   # Smithson & Verkuilen squeeze
    bm = BetaModel(y_pos.to_numpy(), X.loc[pos].to_numpy()).fit(disp=0, maxiter=500)
    k = X.shape[1]
    coefs = pd.DataFrame({"logit P(RR=0)": lm.params.to_numpy(), "p-value (logit)": lm.pvalues.to_numpy(),
                          "Beta regr. (mean)": bm.params[:k], "p-value (Beta)": bm.pvalues[:k]}, index=X.columns)
    return TwoStageModel(lm.params.to_numpy(), np.asarray(bm.params), list(X.columns)), coefs


# --------------------------------------------------------------------------------------
# Validation metrics
# --------------------------------------------------------------------------------------
def cap_gini(y, pred) -> float:
    y, pred = np.asarray(y, float), np.asarray(pred, float)

    def area(order):
        cum = np.cumsum(y[order]) / y.sum()
        return integrate.trapezoid(np.r_[0, cum], np.linspace(0, 1, len(y) + 1)) - 0.5

    return float(area(np.argsort(-pred, kind="stable")) / area(np.argsort(-y, kind="stable")))


def model_metrics(y, pred) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    sp = stats.spearmanr(y, pred).statistic if np.std(pred) > 0 else np.nan
    return {"RMSE": float(np.sqrt(np.mean((y - pred) ** 2))), "MAE": float(np.mean(np.abs(y - pred))),
            "Spearman": float(sp), "CAP Gini": cap_gini(y, pred),
            "mean predicted LGD": float(pred.mean()), "mean observed LGD": float(y.mean())}
