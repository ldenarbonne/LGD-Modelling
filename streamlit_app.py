"""
LGD Modelling Lab – interactive companion to notebooks/lgd_modelling.ipynb

Run locally:  streamlit run streamlit_app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent / "app"))
import lgd_core as core  # noqa: E402

REPO_URL = "https://github.com/ldenarbonne/lgd-modelling"
NOTEBOOK_URL = f"{REPO_URL}/blob/main/notebooks/lgd_modelling.ipynb"
LINKEDIN_URL = "https://www.linkedin.com/in/laurent-motais-de-narbonne/"
SITE_URL = "https://ldenarbonne.github.io/"

C1, C2, C3, C4, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#8a8984"
EST_COLORS = {
    "Beta – moments": C4, "Beta – MLE": C2, "Naive Gaussian kernel": GREY,
    "Gaussian kernel + reflection": C3, "Gaussian kernel on logit": "#e87ba4", "Beta kernel (Chen)": C1,
}
HIST_GREY = "rgba(160,158,150,0.35)"

st.set_page_config(page_title="LGD Modelling Lab", page_icon="📉", layout="wide")


# --------------------------------------------------------------------------------------
# Cached data and models
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading Lending Club recovery data…")
def get_data():
    df = core.load_sample()
    train, test = core.split(df)
    raw = pd.read_csv(core.DATA_PATH, usecols=["default_year", "lgd", "rr"])
    completeness = (raw.dropna().astype({"default_year": int}).groupby("default_year")
                       .agg(defaults=("lgd", "size"), mean_lgd=("lgd", "mean"),
                            share_zero_RR=("rr", lambda s: (s == 0).mean())))
    return df, train, test, completeness.loc[2008:2019]


@st.cache_resource(show_spinner="Fitting the logit + Beta regression benchmark…")
def get_benchmark():
    _, train, _, _ = get_data()
    return core.fit_two_stage(train)


SEGMENTS = {
    "All defaults": lambda d: np.ones(len(d), bool),
    "36-month loans": lambda d: (d.term_months == 36).to_numpy(),
    "60-month loans": lambda d: (d.term_months == 60).to_numpy(),
    "Owners / mortgage": lambda d: (d.owner == 1).to_numpy(),
    "Renters / other": lambda d: (d.owner == 0).to_numpy(),
    "Grades A–B": lambda d: d.grade.isin(["A", "B"]).to_numpy(),
    "Grades C–D": lambda d: d.grade.isin(["C", "D"]).to_numpy(),
    "Grades E–G": lambda d: d.grade.isin(["E", "F", "G"]).to_numpy(),
}


@st.cache_data(show_spinner=False)
def segment_arrays(segment: str):
    _, train, test, _ = get_data()
    f = SEGMENTS[segment]
    rr_tr, rr_te = train.rr.to_numpy()[f(train)], test.rr.to_numpy()[f(test)]
    return rr_tr, rr_te


@st.cache_data(show_spinner=False)
def fit_sample(segment: str, n: int = 8000):
    rr_tr, _ = segment_arrays(segment)
    y = core.continuous_part(rr_tr)
    rng = np.random.default_rng(core.SEED)
    return rng.choice(y, min(n, len(y)), replace=False)


@st.cache_data(show_spinner=False)
def density_on_grid(segment: str, estimator: str, bw: float):
    rr_tr, _ = segment_arrays(segment)
    y = core.continuous_part(rr_tr)
    if estimator == "Beta – moments":
        return stats.beta.pdf(core.GRID, *core.beta_mom(y))
    if estimator == "Beta – MLE":
        return stats.beta.pdf(core.GRID, *core.beta_mle(y))
    fn, _ = core.KERNELS[estimator]
    return fn(core.GRID, fit_sample(segment), bw)


@st.cache_data(show_spinner="Running likelihood cross-validation (about 30–60 seconds)…")
def run_cv(segment: str):
    rr_tr, _ = segment_arrays(segment)
    return core.likelihood_cv(core.continuous_part(rr_tr))


@st.cache_data(show_spinner=False)
def scorecard_results(cfg_key: tuple):
    repaid, mob, term36, owner, nsb, income, n_pools = cfg_key
    cfg = core.ScorecardConfig(list(repaid), list(mob), term36, owner, nsb, list(income), n_pools)
    _, train, test, _ = get_data()
    train, test = train.copy(), test.copy()
    for d in (train, test):
        d["score"] = core.expert_score(d, cfg)
    edges = core.pool_edges(train["score"], cfg.n_pools)
    for d in (train, test):
        d["pool"] = core.assign_pools(d["score"], edges)
    calib = core.calibrate_pools(train)
    for d in (train, test):
        d["lgd_sc"] = d["pool"].map(calib["lgd_lra"]).astype(float)
    model, _ = get_benchmark()
    test["lgd_br"] = model.predict(test)
    cols = ["pool", "score", "lgd", "rr", "lgd_sc", "default_year"]
    return train[cols], test[cols + ["lgd_br"]], calib, edges


@st.cache_data(show_spinner=False)
def pool_density(cfg_key: tuple, pool: str):
    train, _, _, _ = scorecard_results(cfg_key)
    r = train.loc[train.pool == pool, "rr"].to_numpy()
    y = core.continuous_part(r)
    rng = np.random.default_rng(core.SEED)
    sample = rng.choice(y, min(6000, len(y)), replace=False)
    f = core.normalise(core.kde_beta(core.GRID, sample, core.DEFAULT_BW["Beta kernel (Chen)"]))
    return f, float((r == 0).mean()), float((r == 1).mean())


def fmt_pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def style_fig(fig, height=380, hovermode="x unified"):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), hovermode=hovermode,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(showgrid=False)
    return fig


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
df, train, test, completeness = get_data()

with st.sidebar:
    st.title("📉 LGD Modelling Lab")
    st.caption("Interactive companion to the notebook *LGD modelling: Beta distribution vs kernel smoothing vs "
               "expert-judgement scorecard*.")
    st.markdown(f"""
**Data** – Lending Club charged-off loans (Kaggle, CC0), defaults 2008–2015
- {len(df):,} defaults · development {len(train):,} · test {len(test):,}
- RR = net recoveries / EAD, LGD = 1 − RR (undiscounted)

**Links**
- [GitHub repository]({REPO_URL})
- [Full notebook]({NOTEBOOK_URL})
""")
    st.divider()
    st.markdown(f"**Laurent Motais de Narbonne**  \nSenior Risk Manager  \n[LinkedIn]({LINKEDIN_URL}) · [Publications]({SITE_URL})")
    st.caption("Educational project – not a regulatory model.")

tab_data, tab_dens, tab_sc, tab_calc, tab_meth = st.tabs(
    ["📊 Data", "📈 Density estimation", "🧮 Expert scorecard", "🎯 LGD calculator", "📘 Methodology"])

# --------------------------------------------------------------------------------------
# Tab 1 – Data
# --------------------------------------------------------------------------------------
with tab_data:
    st.subheader("Recovery-rate data")
    rr_all = df.rr.to_numpy()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Defaults (2008–2015)", f"{len(df):,}")
    k2.metric("Mean LGD (default-weighted)", fmt_pct(df.lgd.mean()))
    k3.metric("Share with zero recovery", fmt_pct((rr_all == 0).mean()))
    k4.metric("Median RR (continuous part)", fmt_pct(np.median(core.continuous_part(rr_all))))

    c1, c2 = st.columns(2)
    with c1:
        counts, edges_h = np.histogram(rr_all, bins=100, range=(0, 1))
        fig = go.Figure(go.Bar(x=(edges_h[:-1] + edges_h[1:]) / 2, y=counts, marker_color=C1, marker_line_width=0,
                               hovertemplate="RR ≈ %{x:.1%}<br>%{y:,} defaults<extra></extra>"))
        fig.update_layout(title="Recovery-rate distribution", xaxis_title="Recovery rate", yaxis_title="Number of defaults",
                          xaxis_tickformat=".0%", bargap=0.05)
        st.plotly_chart(style_fig(fig, hovermode="closest"), width="stretch")
    with c2:
        by_year = df.groupby("default_year").agg(lgd=("lgd", "mean"), n=("lgd", "size")).reset_index()
        fig = go.Figure(go.Scatter(x=by_year.default_year, y=by_year.lgd, mode="lines+markers", line_color=C2,
                                   customdata=by_year.n, name="Mean LGD",
                                   hovertemplate="%{x}: LGD %{y:.1%} (%{customdata:,} defaults)<extra></extra>"))
        fig.update_layout(title="Mean LGD by default year", xaxis_title="Default year", yaxis_title="Mean LGD",
                          yaxis_tickformat=".0%", xaxis_dtick=1)
        st.plotly_chart(style_fig(fig), width="stretch")

    st.markdown("""
**Reading.** The distribution is *mixed*: a point mass at RR = 0 (no recovery), an almost empty mass at RR = 1, and a
continuous part that is concentrated and **multimodal** around 5–20 %, consistent with charged-off receivables being sold to
debt buyers at bulk prices.
""")
    with st.expander("Why are defaults from 2016 onwards excluded?"):
        st.markdown("The share of zero recoveries jumps for recent default years: those workouts were still open at the "
                    "extraction date (end of 2018). Keeping them would overstate LGD.")
        show = completeness.rename(columns={"defaults": "Defaults", "mean_lgd": "Mean LGD", "share_zero_RR": "Share RR = 0"})
        st.dataframe(show.style.format({"Defaults": "{:,}", "Mean LGD": "{:.1%}", "Share RR = 0": "{:.1%}"}),
                     width="stretch")

# --------------------------------------------------------------------------------------
# Tab 2 – Density estimation
# --------------------------------------------------------------------------------------
with tab_dens:
    st.subheader("Parametric vs kernel estimation of the recovery-rate density")
    st.caption("Point masses at 0 and 1 are kept as discrete probabilities; the estimators below model the continuous "
               "part on (0, 1). Fits use the development sample, metrics the untouched test sample.")

    cc1, cc2, cc3 = st.columns([1.2, 2, 1])
    segment = cc1.selectbox("Segment", list(SEGMENTS), index=0)
    estimators = cc2.multiselect("Estimators", list(EST_COLORS),
                                 default=["Beta – MLE", "Naive Gaussian kernel", "Beta kernel (Chen)"])
    zoom = cc3.radio("Range", ["0–35%", "0–100%"], horizontal=True)

    with st.expander("Bandwidths (defaults = likelihood cross-validation from the notebook)"):
        bw_cols = st.columns(4)
        bws = {}
        for col, (name, (_, grid)) in zip(bw_cols, core.KERNELS.items()):
            options = np.unique(np.r_[np.geomspace(grid[0], grid[-1], 25), core.DEFAULT_BW[name]])
            default = core.DEFAULT_BW[name]
            bws[name] = col.select_slider(name, options=list(options), value=default,
                                          format_func=lambda v: f"{v:.2e}", key=f"bw_{name}")
        st.markdown("For the Beta kernel, the effective smoothing at point *x* is the kernel's standard deviation "
                    "≈ √(b·x(1−x)), so a very small *b* does not mean no smoothing.")
        if st.button("Re-run likelihood cross-validation for this segment"):
            cv = run_cv(segment)
            fig = go.Figure()
            for name, res in cv.items():
                fig.add_trace(go.Scatter(x=res["grid"], y=res["ll"], mode="lines+markers", name=name,
                                         line_color=EST_COLORS[name]))
            fig.update_layout(title="Validation log-likelihood vs bandwidth", xaxis_type="log",
                              xaxis_title="bandwidth (log scale)", yaxis_title="mean log-likelihood")
            st.plotly_chart(style_fig(fig, height=320), width="stretch")
            st.dataframe(pd.Series({k: v["best"] for k, v in cv.items()}, name="best bandwidth")
                         .to_frame().style.format("{:.2e}"), width="stretch")

    rr_tr, rr_te = segment_arrays(segment)
    p0, p1 = float((rr_tr == 0).mean()), float((rr_tr == 1).mean())
    y_tr, y_te = core.continuous_part(rr_tr), core.continuous_part(rr_te)

    if not estimators:
        st.info("Select at least one estimator.")
    else:
        dens = {}
        with st.spinner("Estimating densities…"):
            for e in estimators:
                dens[e] = density_on_grid(segment, e, bws.get(e, 0.0))
        xmax = 0.35 if zoom == "0–35%" else 1.0
        counts, edges_h = np.histogram(y_tr, bins=int(240 * xmax) or 1, range=(0, xmax), density=True)
        counts = counts * ((y_tr <= xmax).mean())
        ymax = st.slider("Cap the vertical axis (density)", 10, 400, 40, step=10,
                         help="The spike of micro-recoveries just above 0 is very high; cap the axis to see the body.")
        fig = go.Figure(go.Bar(x=(edges_h[:-1] + edges_h[1:]) / 2, y=counts, marker_color=HIST_GREY, marker_line_width=0,
                               name="Histogram (development)", hoverinfo="skip"))
        for e, f in dens.items():
            fig.add_trace(go.Scatter(x=core.GRID, y=core.normalise(f), mode="lines", name=e, line=dict(color=EST_COLORS[e], width=2),
                                     hovertemplate=f"{e}<br>RR %{{x:.2%}} · density %{{y:.2f}}<extra></extra>"))
        fig.update_layout(title=f"Density of the continuous part – {segment}", xaxis_title="Recovery rate",
                          yaxis_title="density", xaxis_tickformat=".0%", bargap=0)
        fig.update_xaxes(range=[0, xmax]); fig.update_yaxes(range=[0, ymax])
        st.plotly_chart(style_fig(fig, height=430, hovermode="closest"), width="stretch")

        rows = {e: {"probability mass on [0,1]": float(np.trapezoid(f, core.GRID)),
                    **core.density_metrics(f, y_te, rr_te, p0, p1)} for e, f in dens.items()}
        emp = np.sort(rr_te)
        rows["Empirical (test)"] = {"probability mass on [0,1]": np.nan, "mean log-lik. (test)": np.nan, "KS (test)": 0.0,
                                    "LGD q25": 1 - np.quantile(emp, .75), "LGD median": 1 - np.quantile(emp, .5),
                                    "LGD q75": 1 - np.quantile(emp, .25)}
        tab = pd.DataFrame(rows).T
        st.dataframe(tab.style.format({"probability mass on [0,1]": "{:.3f}", "mean log-lik. (test)": "{:.3f}",
                                       "KS (test)": "{:.3f}", "LGD q25": "{:.1%}", "LGD median": "{:.1%}",
                                       "LGD q75": "{:.1%}"}, na_rep="–")
                     .highlight_min(subset=["KS (test)"], color="rgba(42,120,214,0.18)")
                     .highlight_max(subset=["mean log-lik. (test)"], color="rgba(42,120,214,0.18)"),
                     width="stretch")
        st.caption(f"Segment: {len(rr_tr):,} development / {len(rr_te):,} test defaults · P(RR=0) = {fmt_pct(p0)} · "
                   "LGD quantiles are computed on the full mixed distribution (masses at 0 and 1 + continuous part). "
                   "Higher log-likelihood and lower KS are better.")

# --------------------------------------------------------------------------------------
# Tab 3 – Expert scorecard
# --------------------------------------------------------------------------------------
with tab_sc:
    st.subheader("Expert-judgement LGD scorecard")
    st.caption("Edit the points (more points = higher expected recovery). Pools are cut at score quantiles on the "
               "development sample and calibrated to the long-run average LGD; validation uses the test sample.")
    d0 = core.ScorecardConfig()
    with st.expander("Scorecard points", expanded=True):
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown("**Share of principal repaid**")
            repaid = [st.number_input(lbl, 0, 50, v, key=f"rep{i}")
                      for i, (lbl, v) in enumerate(zip(["< 15%", "15–35%", "35–55%", "≥ 55%"], d0.repaid))]
            st.markdown("**Months on book at default**")
            mob = [st.number_input(lbl, 0, 50, v, key=f"mob{i}")
                   for i, (lbl, v) in enumerate(zip(["≤ 9", "10–18", "> 18"], d0.mob))]
        with s2:
            st.markdown("**Declared annual income**")
            income = [st.number_input(lbl, 0, 50, v, key=f"inc{i}")
                      for i, (lbl, v) in enumerate(zip(["< 40k$", "40–75k$", "≥ 75k$"], d0.income))]
            st.markdown("**Other factors**")
            term36 = st.number_input("36-month term (vs 60)", 0, 50, d0.term36)
            owner = st.number_input("Owner / mortgage (vs renter)", 0, 50, d0.owner)
            nsb = st.number_input("Purpose other than small business", 0, 50, d0.not_small_bus)
        with s3:
            n_pools = st.slider("Number of LGD pools", 3, 8, d0.n_pools)
            if st.button("Reset to the notebook's expert points"):
                for key in list(st.session_state):
                    if key.startswith(("rep", "mob", "inc")):
                        del st.session_state[key]
                st.rerun()
            st.info("Rationale: repaid share and time on book reflect capacity/willingness to pay; owners hold "
                    "attachable assets; small-business loans recover less; income proxies residual repayment capacity.")

    cfg_key = (tuple(repaid), tuple(mob), term36, owner, nsb, tuple(income), n_pools)
    sc_train, sc_test, calib, edges = scorecard_results(cfg_key)

    lra_sorted = calib["lgd_lra"].to_numpy()
    monotonic = bool(np.all(np.diff(lra_sorted) > 0))
    pool_val = (sc_test.groupby("pool", observed=True)
                       .agg(n=("lgd", "size"), observed=("lgd", "mean"), estimated=("lgd_sc", "mean"),
                            sd=("lgd", "std")))
    pool_val["t-stat"] = (pool_val.observed - pool_val.estimated) / (pool_val.sd / np.sqrt(pool_val.n))
    n_sig = int((pool_val["t-stat"].abs() > 1.96).sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("Pools", len(calib))
    (m2.success if monotonic else m2.warning)("Pool LGDs are monotonic" if monotonic else "Pool LGDs are NOT monotonic")
    (m3.success if n_sig == 0 else m3.warning)(f"{n_sig} pool(s) with a significant test gap (|t| > 1.96)")

    c1, c2 = st.columns([1.1, 1])
    with c1:
        se = 1.96 * pool_val.sd / np.sqrt(pool_val.n)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pool_val.index.astype(str), y=pool_val.estimated, mode="markers", name="Scorecard (development LRA)",
                                 marker=dict(symbol="diamond", size=11, color=C1)))
        fig.add_trace(go.Scatter(x=pool_val.index.astype(str), y=pool_val.observed, mode="markers", name="Observed (test) ± 95% CI",
                                 marker=dict(size=10, color=C2), error_y=dict(type="data", array=se, color=C2)))
        fig.update_layout(title="Calibration by pool (test sample)", yaxis_title="Mean LGD", yaxis_tickformat=".1%")
        st.plotly_chart(style_fig(fig, height=380), width="stretch")
    with c2:
        show = calib.join(pool_val[["observed", "t-stat"]].rename(columns={"observed": "observed (test)"}))
        st.markdown("**Pool calibration**")
        st.dataframe(show.style.format({"score_min": "{:.0f}", "score_max": "{:.0f}", "n": "{:,}", "lgd_lra": "{:.2%}",
                                        "MoC": "{:.2%}", "lgd_estimate": "{:.2%}", "observed (test)": "{:.2%}",
                                        "t-stat": "{:.2f}"}), width="stretch")
        st.caption("LRA = default-weighted long-run average (development). MoC = illustrative margin of conservatism "
                   "(bootstrap 90th percentile of the pool mean minus the mean).")

    lra_const = np.full(len(sc_test), sc_train.lgd.mean())
    perf = pd.DataFrame({"Simple mean": core.model_metrics(sc_test.lgd, lra_const),
                         "Expert scorecard": core.model_metrics(sc_test.lgd, sc_test.lgd_sc),
                         "Logit + Beta regression": core.model_metrics(sc_test.lgd, sc_test.lgd_br)}).T
    st.markdown("**Out-of-sample performance (test sample)**")
    st.dataframe(perf.style.format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "Spearman": "{:.3f}", "CAP Gini": "{:.3f}",
                                    "mean predicted LGD": "{:.2%}", "mean observed LGD": "{:.2%}"}, na_rep="–"),
                 width="stretch")
    st.caption("CAP Gini for a continuous target: area between the model's cumulative-loss curve and the diagonal, "
               "relative to the perfect ordering. The benchmark is fitted once on the development sample.")

    with st.expander("Stability over time and benchmark coefficients"):
        stab = (sc_test[sc_test.default_year >= 2011].groupby(["default_year", "pool"], observed=True)
                .lgd.mean().unstack())
        fig = go.Figure()
        pools = list(stab.columns)
        for p, col in zip([pools[0], pools[len(pools) // 2], pools[-1]], [C1, C3, C2]):
            fig.add_trace(go.Scatter(x=stab.index, y=stab[p], mode="lines+markers", name=str(p), line_color=col))
        fig.update_layout(title="Mean LGD by default year – best, middle and worst pools (test, 2011–2015)",
                          yaxis_tickformat=".0%", xaxis_dtick=1)
        st.plotly_chart(style_fig(fig, height=340), width="stretch")
        _, coefs = get_benchmark()
        st.dataframe(coefs.style.format("{:.4f}"), width="stretch")
        st.caption("Note the positive sign of the share of principal repaid on P(RR=0), opposite to its bivariate "
                   "relationship: collinearity with months on book. An expert scorecard avoids such inconsistencies.")

# --------------------------------------------------------------------------------------
# Tab 4 – LGD calculator
# --------------------------------------------------------------------------------------
with tab_calc:
    st.subheader("LGD estimate for a defaulted loan")
    st.caption("Uses the scorecard currently defined in the *Expert scorecard* tab, the Beta-kernel distribution of "
               "the corresponding pool, and the parametric benchmark for comparison.")
    i1, i2, i3 = st.columns(3)
    with i1:
        funded = st.number_input("Funded amount ($)", 1000, 40000, 15000, step=500)
        repaid_pct = st.slider("Share of principal repaid before default", 0.0, 0.95, 0.25, 0.01, format="%.2f")
        mob_in = st.slider("Months on book at default", 0, 60, 14)
    with i2:
        term_in = st.radio("Term", [36, 60], horizontal=True)
        own_in = st.radio("Home ownership", ["Owner / mortgage", "Renter / other"], horizontal=True)
        purpose_in = st.radio("Purpose", ["Other", "Small business"], horizontal=True)
    with i3:
        inc_in = st.number_input("Annual income ($)", 5000, 500000, 60000, step=1000)
        rate_in = st.slider("Interest rate (%)", 5.0, 30.0, 13.0, 0.25)
        dti_in = st.slider("Debt-to-income (%)", 0.0, 45.0, 18.0, 0.5)

    ead = funded * (1 - repaid_pct)
    loan = pd.DataFrame([{
        "pct_principal_repaid": repaid_pct, "mob_at_default": mob_in, "term_months": term_in,
        "term60": int(term_in == 60), "owner": int(own_in.startswith("Owner")), "small_bus": int(purpose_in == "Small business"),
        "annual_inc": inc_in, "log_inc": np.log1p(inc_in), "int_rate": rate_in, "dti": dti_in, "log_ead": np.log(max(ead, 51)),
    }])
    cfg = core.ScorecardConfig(list(repaid), list(mob), term36, owner, nsb, list(income), n_pools)
    score = float(core.expert_score(loan, cfg).iloc[0])
    pool = str(core.assign_pools(pd.Series([score]), edges).iloc[0])
    lgd_lra, moc = float(calib.loc[pool, "lgd_lra"]), float(calib.loc[pool, "MoC"])
    model, _ = get_benchmark()
    lgd_bench = float(model.predict(loan)[0])

    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Scorecard score", f"{score:.0f} / {cfg.max_score}")
    o2.metric("LGD pool", pool)
    o3.metric("Pool LGD (LRA + MoC)", fmt_pct(lgd_lra + moc), help=f"LRA {fmt_pct(lgd_lra, 2)} + MoC {fmt_pct(moc, 2)}")
    o4.metric("Benchmark LGD (logit + Beta regr.)", fmt_pct(lgd_bench))

    f_pool, p0_pool, p1_pool = pool_density(cfg_key, pool)
    F = core.cdf_from_density(f_pool)
    q = {lab: 1 - core.mixed_rr_quantile(1 - lv, F, p0_pool, p1_pool)
         for lab, lv in [("median", 0.50), ("75th pct.", 0.75), ("80th pct.", 0.80)]}

    c1, c2 = st.columns([1.6, 1])
    with c1:
        lgd_grid = 1 - core.GRID[::-1]
        fig = go.Figure(go.Scatter(x=lgd_grid, y=f_pool[::-1] * (1 - p0_pool - p1_pool), mode="lines", fill="tozeroy",
                                   line_color=C1, name="continuous part",
                                   hovertemplate="LGD %{x:.1%} · density %{y:.2f}<extra></extra>"))
        fig.add_vline(x=lgd_lra, line_dash="dash", line_color=C2, annotation_text="pool LRA", annotation_position="top left")
        fig.update_layout(title=f"Pool {pool} – LGD distribution (Beta kernel), plus {fmt_pct(p0_pool)} at LGD = 100%",
                          xaxis_title="LGD", yaxis_title="density (scaled by continuous share)", xaxis_tickformat=".0%")
        fig.update_xaxes(range=[0.6, 1.0]); fig.update_yaxes(range=[0, 40])
        st.plotly_chart(style_fig(fig, height=380, hovermode="closest"), width="stretch")
    with c2:
        st.markdown("**Pool LGD distribution**")
        st.dataframe(pd.DataFrame({"value": [fmt_pct(lgd_lra, 2), fmt_pct(p0_pool), *[fmt_pct(v) for v in q.values()],
                                             f"${ead * (lgd_lra + moc):,.0f}"]},
                                  index=["Mean LGD (LRA)", "P(LGD = 100%)", *[f"LGD {k}" for k in q],
                                         f"Estimated loss (LGD × EAD of ${ead:,.0f})"]), width="stretch")
        st.caption("Quantiles combine the point mass at RR = 0 with the Beta-kernel density of the pool's continuous "
                   "part. EAD = funded amount × (1 − share repaid).")

# --------------------------------------------------------------------------------------
# Tab 5 – Methodology
# --------------------------------------------------------------------------------------
with tab_meth:
    st.subheader("Methodology and limitations")
    st.markdown(rf"""
**Target.** $EAD = \text{{funded}} - \text{{principal repaid}}$, $RR = (\text{{recoveries}} - \text{{collection fees}})/EAD$,
$LGD = 1 - RR \in [0, 1]$ – a simplified, undiscounted workout LGD.

**Parametric approach.** Beta distribution fitted by method of moments and maximum likelihood; zero-one-inflated Beta
$F(x) = p_0\,\mathbb 1_{{x\ge0}} + (1-p_0-p_1)\,F_{{\mathcal B}}(x) + p_1\,\mathbb 1_{{x\ge1}}$. Conditional benchmark: logit on
$P(RR=0)$ and Beta regression (Ferrari & Cribari-Neto, 2004) on the continuous part,
$\widehat{{LGD}} = \hat p_0 + (1-\hat p_0)(1-\hat\mu)$.

**Kernel smoothing.** Density of the continuous part estimated without shape assumption. Naive Gaussian kernels leak
mass outside [0,1]; reflection and logit transforms correct this; the **Beta kernel** of Chen (1999),
$\hat f(x)=\frac1n\sum_i \mathrm{{Beta}}(X_i;\,x/b+1,\,(1-x)/b+1)$, is free of boundary bias and is the approach of
Renault & Scaillet (2004) for recovery-rate densities. Bandwidths are chosen by likelihood cross-validation.

**Expert scorecard.** Factors and points set *a priori* from economic rationale, challenged against bivariate evidence;
pools cut at score quantiles; pool LGD = long-run average + illustrative margin of conservatism.

**Main findings.** The Beta distribution reproduces the mean LGD but not the multimodal shape (KS ≈ 0.27 vs 0.008 for
the Beta kernel). Discriminatory power is weak for all conditional models (recoveries largely pooled through debt
sales); the scorecard is monotonic and well calibrated, the parametric benchmark ranks slightly better but shows a
calibration bias and sign reversals.

**Limitations.** Undiscounted LGD (no recovery cash-flow dates); indirect costs not allocated; incomplete workouts handled
by exclusion; US unsecured consumer loans sold to debt buyers – not transferable as such to collateralised European
portfolios; the *downturn* analysis in the notebook is illustrative only (EBA/GL/2019/03).

**References.** Renault & Scaillet (2004), *Journal of Banking & Finance* 28(12) · Chen (1999), *CSDA* 31 ·
Ferrari & Cribari-Neto (2004), *J. Applied Statistics* 31 · EBA/GL/2017/16 · EBA/GL/2019/03.

Source code and full analysis: [{REPO_URL}]({REPO_URL})
""")
