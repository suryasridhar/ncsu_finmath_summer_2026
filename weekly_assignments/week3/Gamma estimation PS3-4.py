#!/usr/bin/env python
# coding: utf-8

# In[7]:


import zipfile
with zipfile.ZipFile("Options-20260617T172821Z-3-001.zip", 'r') as z:
    z.extractall(".")


# In[ ]:


# AS Model Gamma Estimation — Multi-Hour (v2)
# NCSU Summer 2026 GLFT Project
#
# Method:
#   1. Load 15+ hours of options OB data for the most liquid symbol
#   2. Apply latency filter to trades (recv_ts_ms - time <= 60s)
#      to remove stale websocket-replay entries
#   3. Carry-forward real trade inventory via merge_asof
#   4. Estimate γ using OB size imbalance as primary inventory proxy
#      (real trades too sparse/small to drive meaningful γ estimates)
#   5. Back-calculate reservation price adjustment as sanity check
#
# Symbol chosen: BTCUSD-31JUL26-62000-P
#   - BTCUSD-3JUL26-65000-C (original) had 0 live trades in 15 hours
#   - BTCUSD-31JUL26-62000-P had 16 live trades — most liquid available
#
# Known limitation:
#   γ magnitude depends on σ² unit convention (per-tick pct vs annualised $).
#   With per-tick pct σ² ≈ 1.3e-5 and T-t ≈ 0.12 yr, the denominator is ~1e-6,
#   inflating γ to ~1000–2000. The implied inventory risk premium (q·γ·σ²·T-t)
#   is only ~$0.03–3, << half-spread of ~$20, suggesting the market maker's
#   quote adjustment from inventory is small relative to their spread — consistent
#   with an illiquid options market where each trade is only 0.01 BTC.


# In[14]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OPTIONS_DIR = "Options"
TARGET_SYM = "BTCUSD-31JUL26-62000-P"
T_EXPIRY = pd.Timestamp("2026-07-31", tz="UTC")
MAX_TRADE_AGE_MS = 60_000

# load order book snapshots across all hours
ob_parts = []
hour_dirs = sorted(
    d for d in os.listdir(OPTIONS_DIR)
    if os.path.isdir(os.path.join(OPTIONS_DIR, d)) and not d.startswith(".")
)
for h in hour_dirs:
    ob_file = os.path.join(OPTIONS_DIR, h, f"{TARGET_SYM}.parquet")
    if not os.path.exists(ob_file):
        continue
    t = pd.read_parquet(ob_file)
    for col in ["bid1_px", "ask1_px", "bid1_sz", "ask1_sz", "recv_ts_ms"]:
        if col in t.columns:
            t[col] = pd.to_numeric(t[col], errors="coerce")
    ob_parts.append(t)

ob = pd.concat(ob_parts, ignore_index=True)
ob = ob.dropna(subset=["bid1_px", "ask1_px"]).reset_index(drop=True)
ob["recv_dt"] = pd.to_datetime(ob["recv_ts_ms"], unit="ms", utc=True)
ob = ob.sort_values("recv_ts_ms").reset_index(drop=True)
ob["mid"] = (ob["bid1_px"] + ob["ask1_px"]) / 2
ob["spread"] = ob["ask1_px"] - ob["bid1_px"]

print(f"Total OB rows: {len(ob):,}")
print(f"Time range: {ob['recv_dt'].iloc[0]} → {ob['recv_dt'].iloc[-1]}")
print(f"Mid range: {ob['mid'].min():.1f} – {ob['mid'].max():.1f} USD")
print(f"Avg spread: {ob['spread'].mean():.1f} USD")

# filter trades by latency to remove stale websocket replay entries
tr_parts = []
for h in hour_dirs:
    tf = os.path.join(OPTIONS_DIR, h, "trades.parquet")
    if not os.path.exists(tf):
        continue
    t = pd.read_parquet(tf)
    t["latency_ms"] = t["recv_ts_ms"] - t["time"]
    live = t[(t["latency_ms"] <= MAX_TRADE_AGE_MS) & (t["symbol"] == TARGET_SYM)].copy()
    raw = (t["symbol"] == TARGET_SYM).sum()
    print(f"  [{h}] raw={raw}  live={len(live)}")
    tr_parts.append(live)

trades = (pd.concat(tr_parts, ignore_index=True)
          .drop_duplicates(subset=["time"])
          .sort_values("recv_ts_ms")
          .reset_index(drop=True))
trades["recv_dt"] = pd.to_datetime(trades["recv_ts_ms"], unit="ms", utc=True)

print(f"Total live trades: {len(trades)}")
if len(trades):
    trades["signed_qty"] = trades.apply(
        lambda r: r["qty"] if r["trade_side"] == 2 else -r["qty"], axis=1
    )
    trades["q_trade"] = trades["signed_qty"].cumsum()
    print(trades[["recv_dt", "price", "qty", "trade_side", "latency_ms", "q_trade"]]
          .to_string(index=False))

# merge trade inventory into OB snapshots via carry-forward
if len(trades):
    ob = pd.merge_asof(
        ob.sort_values("recv_ts_ms"),
        trades[["recv_ts_ms", "q_trade"]].sort_values("recv_ts_ms"),
        on="recv_ts_ms", direction="backward",
    )
    ob["q_trade"] = ob["q_trade"].fillna(0)
else:
    ob["q_trade"] = 0.0

# OB size imbalance as inventory proxy (rolling 100 snapshots)
ob["bid1_sz"] = pd.to_numeric(ob["bid1_sz"], errors="coerce").fillna(0)
ob["ask1_sz"] = pd.to_numeric(ob["ask1_sz"], errors="coerce").fillna(0)
ob["q_ob"] = (ob["bid1_sz"] - ob["ask1_sz"]).rolling(100, min_periods=1).sum()

# compute σ² and time-to-expiry
sigma2 = ob["mid"].pct_change().dropna().var()
print(f"σ² = {sigma2:.8f}")

ob["T_minus_t"] = (T_EXPIRY - ob["recv_dt"]).dt.total_seconds() / (365.25 * 24 * 3600)
ob["S_fair"] = ob["mid"].rolling(50, center=True, min_periods=10).mean()
ob["r"] = ob["mid"]

# estimate γ from AS reservation price equation: r = S - q·γ·σ²·(T-t)
mask = ob["q_ob"].abs() > 0.5
dv = ob[mask].copy()
dv["gamma_t"] = (dv["S_fair"] - dv["r"]) / (dv["q_ob"] * sigma2 * dv["T_minus_t"])
q99 = dv["gamma_t"].abs().quantile(0.99)
dv = dv[dv["gamma_t"].abs() <= q99]

gamma_mean = dv["gamma_t"].mean()
gamma_median = dv["gamma_t"].median()

print(f"Mean γ = {gamma_mean:.1f}")
print(f"Median γ = {gamma_median:.1f}")

ob["inv_risk_premium"] = ob["q_ob"] * gamma_median * sigma2 * ob["T_minus_t"]
print(f"Inventory risk premium median = {ob['inv_risk_premium'].abs().median():.4f} USD")
print(f"Half-spread median = {(ob['spread']/2).median():.2f} USD")
print(f"Ratio = {ob['inv_risk_premium'].abs().median()/(ob['spread']/2).median():.4f}")

# plots: mid price, trade inventory, OB imbalance, γ over time
fig, axes = plt.subplots(4, 1, figsize=(14, 14))

axes[0].plot(ob["recv_dt"], ob["mid"], color="steelblue", lw=0.6, label="mid")
axes[0].plot(ob["recv_dt"], ob["S_fair"], color="orange", lw=1.2, ls="--", label="S_fair (rolling 50)")
if len(trades):
    for _, row in trades.iterrows():
        c = "lime" if row["trade_side"] == 2 else "red"
        axes[0].axvline(row["recv_dt"], color=c, alpha=0.5, lw=1.5)
    axes[0].scatter([], [], color="lime", label="live trade: sell (side=2)")
    axes[0].scatter([], [], color="red", label="live trade: buy (side=1)")
axes[0].set_title(f"{TARGET_SYM} | Mid-Price (15+ hours, {len(trades)} live trades)")
axes[0].set_ylabel("Option Price ($)")
axes[0].legend(fontsize=8)
axes[0].grid(True)

axes[1].step(ob["recv_dt"], ob["q_trade"], color="tomato", lw=1.2, where="post",
             label="q_trade (real trades, carry-forward)")
axes[1].axhline(0, color="gray", ls="--")
axes[1].set_title(f"Inventory — Trade-Based ({len(trades)} live trades, max q={ob['q_trade'].max():.2f})")
axes[1].set_ylabel("q (BTC)")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(ob["recv_dt"], ob["q_ob"], color="green", lw=0.6,
             label="q_ob (rolling 100-snapshot OB imbalance)")
axes[2].axhline(0, color="gray", ls="--")
axes[2].set_title("Inventory Proxy — OB Size Imbalance (rolling 100 snapshots)")
axes[2].set_ylabel("q")
axes[2].legend()
axes[2].grid(True)

axes[3].plot(dv["recv_dt"], dv["gamma_t"], color="purple", lw=0.5, alpha=0.7)
axes[3].axhline(gamma_mean, color="red", ls="--", label=f"Mean γ = {gamma_mean:.0f}")
axes[3].axhline(gamma_median, color="orange", ls="--", label=f"Median γ = {gamma_median:.0f}")
axes[3].set_title("Estimated Risk-Aversion γ (AS Model, OB imbalance q)")
axes[3].set_ylabel("γ")
axes[3].legend()
axes[3].grid(True)

plt.tight_layout()
plt.savefig("as_gamma_v2_final.png", dpi=150)
plt.show()

dv[["recv_dt", "mid", "S_fair", "q_ob", "q_trade", "T_minus_t", "gamma_t"]].to_csv(
    "as_gamma_v2_results.csv", index=False
)
print("Plot saved: as_gamma_v2_final.png")
print("Results saved: as_gamma_v2_results.csv")

# calibrate simulator parameters from data
N_OB = max(len(ob), 1)
N_TRADES = max(len(trades), 1)
lambda0_per_tick = N_TRADES / N_OB
half_spread_med = (ob["spread"] / 2).median()
kappa = np.log(2) / max(half_spread_med, 1e-6)
T_SIM = ob["T_minus_t"].median()
S0 = ob["mid"].median()

# AS market-maker simulator: quotes bid/ask around reservation price,
# fills arrive as Poisson process with rate λ(δ) = λ0·exp(-κ·δ)
def simulate_as(gamma, sigma2, T, S0, lambda0, kappa, n_steps=2000, seed=None):
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(sigma2)
    S = float(S0)
    q = 0.0
    cash = 0.0
    pnl_arr = np.empty(n_steps)
    q_arr = np.empty(n_steps)
    for i in range(n_steps):
        t_rem = max(T * (1.0 - i / n_steps), 1e-8)
        r = S - q * gamma * sigma2 * t_rem
        half_delta = (gamma * sigma2 * t_rem) / 2.0 + np.log1p(gamma / kappa) / gamma
        half_delta = max(half_delta, 0.0)
        bid_px = r - half_delta
        ask_px = r + half_delta
        p_bid = 1.0 - np.exp(-lambda0 * np.exp(-kappa * max(S - bid_px, 0.0)))
        p_ask = 1.0 - np.exp(-lambda0 * np.exp(-kappa * max(ask_px - S, 0.0)))
        if rng.random() < p_bid:
            q += 1.0
            cash -= bid_px
        if rng.random() < p_ask:
            q -= 1.0
            cash += ask_px
        S *= (1.0 + sigma * rng.standard_normal())
        pnl_arr[i] = cash + q * S
        q_arr[i] = q
    pnl_diff = np.diff(pnl_arr)
    std_diff = pnl_diff.std()
    sharpe = (pnl_diff.mean() / std_diff * np.sqrt(n_steps)) if std_diff > 1e-12 else 0.0
    return {
        "pnl_series": pnl_arr,
        "q_series": q_arr,
        "sharpe": sharpe,
        "inv_variance": float(q_arr.var()),
        "final_pnl": float(pnl_arr[-1]),
    }

# sweep γ over 3 orders of magnitude, run 30 MC sims per point
log_centre = np.log10(max(abs(gamma_median), 1.0))
gammas = np.logspace(log_centre - 2, log_centre + 1, 40)
N_SIMS = 30
N_STEPS = 2000

rows = []
for idx, g in enumerate(gammas):
    sharpes = []
    inv_vars = []
    final_pnls = []
    for s in range(N_SIMS):
        res = simulate_as(
            gamma=g, sigma2=sigma2, T=T_SIM, S0=S0,
            lambda0=lambda0_per_tick, kappa=kappa,
            n_steps=N_STEPS, seed=idx * N_SIMS + s,
        )
        sharpes.append(res["sharpe"])
        inv_vars.append(res["inv_variance"])
        final_pnls.append(res["final_pnl"])
    rows.append({
        "gamma": g,
        "log10_gamma": np.log10(g),
        "sharpe_mean": np.mean(sharpes),
        "sharpe_std": np.std(sharpes),
        "inv_variance_mean": np.mean(inv_vars),
        "inv_variance_std": np.std(inv_vars),
        "final_pnl_mean": np.mean(final_pnls),
        "final_pnl_std": np.std(final_pnls),
    })
    if (idx + 1) % 8 == 0 or idx == len(gammas) - 1:
        print(f"[{idx+1}/{len(gammas)}] γ={g:.1f}  Sharpe={np.mean(sharpes):+.3f}  InvVar={np.mean(inv_vars):.3f}")

sweep_df = pd.DataFrame(rows)
sweep_df.to_csv("gamma_sensitivity.csv", index=False)

# identify most effective γ regime by three criteria
best_sharpe_idx = sweep_df["sharpe_mean"].idxmax()
gamma_best_sharpe = sweep_df.loc[best_sharpe_idx, "gamma"]

inv_max = sweep_df["inv_variance_mean"].max()
knee_rows = sweep_df[sweep_df["inv_variance_mean"] <= 0.10 * inv_max]
gamma_knee = knee_rows["gamma"].iloc[0] if len(knee_rows) else gamma_best_sharpe

eff = sweep_df["sharpe_mean"] / (sweep_df["inv_variance_mean"] + 1e-12)
best_eff_idx = eff.idxmax()
gamma_best_eff = sweep_df.loc[best_eff_idx, "gamma"]

tercile = len(gammas) // 3
regime = ("low" if best_eff_idx < tercile else "mid" if best_eff_idx < 2 * tercile else "high")

print(f"Best Sharpe γ = {gamma_best_sharpe:.1f}")
print(f"InvVar knee γ = {gamma_knee:.1f}")
print(f"Best efficiency γ = {gamma_best_eff:.1f}")
print(f"Most effective regime: {regime.upper()}")

fig, axes = plt.subplots(3, 1, figsize=(11, 12))

ax = axes[0]
ax.semilogx(sweep_df["gamma"], sweep_df["sharpe_mean"], color="royalblue", lw=2, label="Mean Sharpe (30 sims)")
ax.fill_between(sweep_df["gamma"],
                sweep_df["sharpe_mean"] - sweep_df["sharpe_std"],
                sweep_df["sharpe_mean"] + sweep_df["sharpe_std"],
                alpha=0.20, color="royalblue", label="±1 std")
ax.axvline(gamma_best_sharpe, color="red", ls="--", label=f"Best Sharpe γ={gamma_best_sharpe:,.0f}")
ax.axvline(gamma_median, color="orange", ls=":", lw=1.5, label=f"Data estimate γ={gamma_median:,.0f}")
ax.set_xlabel("γ (log scale)")
ax.set_ylabel("Sharpe Ratio")
ax.set_title("γ Sensitivity — Sharpe Ratio vs γ")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.4)

ax = axes[1]
ax.semilogx(sweep_df["gamma"], sweep_df["inv_variance_mean"], color="seagreen", lw=2, label="Mean Inv. Variance")
ax.fill_between(sweep_df["gamma"],
                sweep_df["inv_variance_mean"] - sweep_df["inv_variance_std"],
                sweep_df["inv_variance_mean"] + sweep_df["inv_variance_std"],
                alpha=0.20, color="seagreen")
ax.axvline(gamma_knee, color="red", ls="--", label=f"InvVar knee γ={gamma_knee:,.0f}")
ax.axvline(gamma_median, color="orange", ls=":", lw=1.5, label=f"Data estimate γ={gamma_median:,.0f}")
ax.set_xlabel("γ (log scale)")
ax.set_ylabel("Var(q)")
ax.set_title("γ Sensitivity — Inventory Variance vs γ")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.4)

ax = axes[2]
ax.semilogx(sweep_df["gamma"], eff, color="darkorchid", lw=2, label="Sharpe / Inventory Variance")
ax.axvline(gamma_best_eff, color="red", ls="--",
           label=f"Best efficiency γ={gamma_best_eff:,.0f} [{regime.upper()} regime]")
ax.axvline(gamma_median, color="orange", ls=":", lw=1.5, label=f"Data estimate γ={gamma_median:,.0f}")
ax.set_xlabel("γ (log scale)")
ax.set_ylabel("Sharpe / Var(q)")
ax.set_title(f"γ Sensitivity — Efficiency | Most effective regime: {regime.upper()} (γ={gamma_best_eff:,.0f})")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.4)

plt.tight_layout()
plt.savefig("gamma_sensitivity.png", dpi=150)
plt.show()


# In[ ]:




