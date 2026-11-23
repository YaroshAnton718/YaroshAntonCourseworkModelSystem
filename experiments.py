from __future__ import annotations
import csv
import math
import os
import statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import nnls
from model_d7 import D7Params, replace, replicate
from verification import ci95, variance_simple

plt.rcParams.update({"figure.dpi": 110, "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 11,
                     "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10})
LABEL_LAM = "Інтенсивність надходження lambda, 1/с"

def _save_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def _area_N(res_run):
    return np.array([c[1]["ARR"] - c[1]["Done"] for c in res_run["checkpoints"]])

def transient_study(out_dir, cfg, workers=None):
    p = D7Params()
    n_runs, step, t_total, eps = cfg["tr_runs"], cfg["tr_step"], cfg["tr_total"], cfg["tr_eps"]
    res = replicate(p, n_runs, t_total, 0.0, seed0=1, workers=workers, checkpoint_step=step)
    ts = np.array([c[0] for c in res[0]["checkpoints"]])
    areas = np.array([_area_N(r) for r in res])
    ybar = areas / ts
    spread = ybar.max(axis=0) - ybar.min(axis=0)
    ok = spread < eps
    T_run = None
    for k in range(len(ts)):
        if ok[k:].all():
            T_run = float(ts[k])
            break

    wres = replicate(p, cfg["wl_runs"], cfg["wl_total"], 0.0, seed0=300, workers=workers,
                     checkpoint_step=cfg["wl_step"])
    wts = np.array([c[0] for c in wres[0]["checkpoints"]])
    wareas = np.array([_area_N(r) for r in wres])
    win = np.diff(np.vstack([np.zeros((1, len(wres))), wareas.T]), axis=0).T / cfg["wl_step"]
    avg = win.mean(axis=0)
    w = cfg["wl_smooth"]
    kern = np.ones(w) / w
    smooth = np.convolve(avg, kern, mode="valid")
    sx = wts[w - 1:] - cfg["wl_step"] * (w - 1) / 2.0
    steady = float(ybar[:, -1].mean())
    half = len(smooth) // 2
    noise = float(smooth[half:].std())
    tol = max(cfg["wl_delta"] * steady, 3.0 * noise)
    ok_w = np.abs(smooth - steady) <= tol
    welch = float(sx[-1])
    for k in range(len(sx)):
        if ok_w[k:].all():
            welch = float(sx[k])
            break
    welch = max(welch, 0.0)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for i in range(n_runs):
        ax[0].plot(ts, ybar[i], lw=0.9, label=f"прогін {i + 1}")
    if T_run:
        ax[0].axvline(T_run, color="k", ls="--", label=f"T = {T_run:.0f} с")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Модельний час t, с")
    ax[0].set_ylabel("Середня кількість запитів y(t), шт.")
    ax[0].set_title("Збіжність середнього за прогонами")
    ax[0].legend(fontsize=9)
    ax[1].plot(wts, avg, color="0.7", label="середнє за %d прогонами, вікно %d с" % (cfg["wl_runs"], cfg["wl_step"]))
    ax[1].plot(sx, smooth, "r", label="ковзне середнє (%d вікон)" % w)
    ax[1].axhline(steady, color="g", ls=":", label="стале значення")
    ax[1].axvline(welch, color="k", ls="--", label=f"перехідний період ~ {welch:.0f} с")
    ax[1].set_xlabel("Модельний час t, с")
    ax[1].set_ylabel("Кількість запитів у системі N, шт.")
    ax[1].set_title("Перехідний період (метод Велча)")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "transient.png"))
    plt.close(fig)
    _save_csv(os.path.join(out_dir, "transient.csv"), ["t, с"] + [f"y{i + 1}" for i in range(n_runs)] + ["spread"],
              [[ts[k]] + list(ybar[:, k]) + [spread[k]] for k in range(len(ts))])
    return {"eps": eps, "T_run": T_run, "welch": welch, "steady_N": float(steady),
            "final_spread": float(spread[-1]), "tr_total": t_total}

def number_of_runs(cfg, warmup, t_run, workers=None):
    pilot = replicate(D7Params(), cfg["pilot_runs"], warmup + t_run, warmup, seed0=100, workers=workers)
    vals = [r["W"] for r in pilot]
    sigma = math.sqrt(variance_simple(vals))
    eps, beta = cfg["run_eps"], 0.95
    n = math.ceil(sigma ** 2 / (eps ** 2 * (1 - beta)))
    return {"sigma_run": sigma, "eps": eps, "beta": beta, "n_required": n, "pilot_mean": st.fmean(vals)}

def chi2_test(x, cdf, n_params, k_bins=None):
    x = np.asarray(x)
    n = len(x)
    k = k_bins or int(round(2 * n ** 0.4))
    edges = np.linspace(x.min(), x.max(), k + 1)
    obs, _ = np.histogram(x, bins=edges)
    p = np.diff(cdf(edges))
    p[0] += cdf(edges[0])
    p[-1] += 1.0 - cdf(edges[-1])
    exp = p * n
    O, E, o_acc, e_acc = [], [], 0.0, 0.0
    for o, e in zip(obs, exp):
        o_acc += o
        e_acc += e
        if e_acc >= 5:
            O.append(o_acc)
            E.append(e_acc)
            o_acc = e_acc = 0.0
    if e_acc > 0:
        O[-1] += o_acc
        E[-1] += e_acc
    O, E = np.array(O), np.array(E)
    chi2 = float(((O - E) ** 2 / E).sum())
    df = len(O) - 1 - n_params
    pval = float(1 - stats.chi2.cdf(chi2, df)) if df > 0 else float("nan")
    return chi2, df, pval, len(O)

def statistics_and_distribution(out_dir, cfg, warmup, t_run, n_cheb, workers=None):
    p = D7Params()
    n_runs = max(n_cheb, cfg["dist_runs"])
    res = replicate(p, n_runs, warmup + t_run, warmup, seed0=2000, workers=workers)
    x = np.array([r["W"] for r in res])
    out = {"n_runs": n_runs, "T_run": t_run, "warmup": warmup}
    out["W_mean"], out["W_ci"] = ci95(x.tolist())
    out["W_std"] = math.sqrt(variance_simple(x.tolist()))
    out["W_min"], out["W_max"] = float(x.min()), float(x.max())
    for key in ("N", "X", "rho1", "rho2", "Lq", "Wq"):
        out[key] = ci95([r[key] for r in res])
    out["inv"] = all(r["invariants"] for r in res)

    cands = {}
    mu = float(x.mean())
    sd = math.sqrt(variance_simple(x.tolist()))
    cands["нормальний"] = (lambda v: stats.norm.cdf(v, mu, sd), 2, f"m={mu:.2f}, s={sd:.2f}",
                           lambda v: stats.norm.pdf(v, mu, sd))
    s_, loc, sc = stats.lognorm.fit(x, floc=0)
    cands["логнормальний"] = (lambda v: stats.lognorm.cdf(v, s_, loc, sc), 2, f"s={s_:.3f}, scale={sc:.2f}",
                              lambda v: stats.lognorm.pdf(v, s_, loc, sc))
    a, loc_g, sc_g = stats.gamma.fit(x, floc=0)
    cands["Ерланга/гамма"] = (lambda v: stats.gamma.cdf(v, a, loc_g, sc_g), 2, f"k={a:.2f}, scale={sc_g:.3f}",
                              lambda v: stats.gamma.pdf(v, a, loc_g, sc_g))
    c_, loc_w, sc_w = stats.weibull_min.fit(x, floc=0)
    cands["Вейбулла"] = (lambda v: stats.weibull_min.cdf(v, c_, loc_w, sc_w), 2, f"k={c_:.2f}, scale={sc_w:.2f}",
                         lambda v: stats.weibull_min.pdf(v, c_, loc_w, sc_w))
    table = []
    for name, (cdf, k, pars, _) in cands.items():
        chi2, df, pv, bins = chi2_test(x, cdf, k)
        table.append((name, pars, chi2, df, pv))
    out["dist_table"] = table
    out["bins"] = bins
    ok = [t for t in table if not math.isnan(t[4]) and t[4] > 0.05]
    out["best_dist"] = max(ok, key=lambda t: t[4])[0] if ok else max(table, key=lambda t: t[4] if t[4] == t[4] else -1)[0]
    out["skew"], out["kurt"] = float(stats.skew(x)), float(stats.kurtosis(x))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    k_hist = int(round(2 * len(x) ** 0.4))
    ax.hist(x, bins=k_hist, density=False, alpha=0.55, label="гістограма частот")
    width = (x.max() - x.min()) / k_hist
    grid = np.linspace(x.min(), x.max(), 300)
    for name, (_, _, _, pdf) in cands.items():
        ax.plot(grid, pdf(grid) * len(x) * width, lw=1.6, label=name)
    ax.set_xlabel("Середній час обробки запиту за прогін W, с")
    ax.set_ylabel("Кількість прогонів (частота)")
    ax.set_title(f"Гістограма частот W (вибірка {len(x)} незалежних прогонів)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "distribution.png"))
    plt.close(fig)
    _save_csv(os.path.join(out_dir, "distribution_tests.csv"),
              ["закон", "параметри", "chi2", "ступені свободи", "p-value"], table)
    np.savetxt(os.path.join(out_dir, "sample_W.csv"), x, fmt="%.5f", header="W, с", encoding="utf-8")
    return out

def normalize(x, lo, hi):
    x0, d = (hi + lo) / 2.0, (hi - lo) / 2.0
    return (np.asarray(x) - x0) / d

def least_squares(F, y):
    return np.linalg.solve(F.T @ F, F.T @ y)

def anova(groups):
    g = np.array(groups)
    q, n = g.shape
    grand = g.mean()
    d_f = n * ((g.mean(axis=1) - grand) ** 2).sum() / (q - 1)
    d_r = ((g - g.mean(axis=1, keepdims=True)) ** 2).sum() / (q * (n - 1))
    F = 0.0 if d_r == 0.0 and d_f == 0.0 else (float("inf") if d_r == 0.0 else d_f / d_r)
    means = g.mean(axis=1)
    return {"q": q, "n": n, "D_factor": d_f, "D_resid": d_r, "F": F,
            "range": float(means.max() - means.min()),
            "F_crit": stats.f.ppf(0.95, q - 1, q * (n - 1)),
            "p": 1.0 if F == 0.0 else float(1 - stats.f.cdf(F, q - 1, q * (n - 1)))}

def intensity_study(out_dir, cfg, warmup, t_run, workers=None):
    base = D7Params()
    lams = np.linspace(cfg["lam_lo"], cfg["lam_hi"], cfg["lam_levels"])
    rows, groups, groups_wq = [], [], []
    for lam in lams:
        p = replace(base, interarrival_mean=1.0 / lam, interarrival_half=0.3 / lam)
        res = replicate(p, cfg["intens_runs"], warmup + t_run, warmup, seed0=7000, workers=workers)
        w = [r["W"] for r in res]
        wq = [r["Wq"] for r in res]
        m, h = ci95(w)
        mq, hq = ci95(wq)
        groups.append(w)
        groups_wq.append(wq)
        rows.append([lam, 1.0 / lam, m, h, st.fmean(r["rho1"] for r in res), st.fmean(r["rho2"] for r in res),
                     st.fmean(r["N"] for r in res), mq, hq])
    _save_csv(os.path.join(out_dir, "intensity.csv"),
              ["lambda, 1/с", "середній інтервал, с", "W, с", "+-CI95, с", "rho1", "rho2", "N, шт.",
               "Wq, с", "+-CI95 Wq, с"], rows)
    lam_arr = np.array([r[0] for r in rows])
    y = np.array([r[2] for r in rows])
    y_wq = np.array([r[7] for r in rows])
    tp = base.p_found * base.search_mean

    def poly(l):
        x = normalize(l, lam_arr.min(), lam_arr.max())
        return np.column_stack([np.ones_like(x), x, x ** 2])

    def f_rho(l):
        r = l * tp
        return r / (1.0 - r)

    def solve(F, resp, nonneg, free_intercept):
        if not nonneg:
            return least_squares(F, resp)
        if free_intercept:
            A = np.column_stack([F[:, 0], -F[:, 0], F[:, 1:]])
            b = nnls(A, resp)[0]
            return np.concatenate([[b[0] - b[1]], b[2:]])
        return nnls(F, resp)[0]

    def fit_models(resp, for_wq):
        queue_w = lambda l: np.column_stack([np.ones_like(l), f_rho(l)])
        queue_q = lambda l: np.column_stack([f_rho(l)])
        specs = [("поліном 2-го степеня від x", poly, False),
                 ("b1*f(rho), b1 >= 0" if for_wq else "b0 + b1*f(rho), bi >= 0", queue_q if for_wq else queue_w, True)]
        out = []
        for name, f, nonneg in specs:
            b = solve(f(lam_arr), resp, nonneg, not for_wq)
            fit = f(lam_arr) @ b
            r2 = 1.0 - ((resp - fit) ** 2).sum() / ((resp - resp.mean()) ** 2).sum()
            n_pts, m = len(resp), len(b)
            r_idx = math.sqrt(max(0.0, r2))
            r_adj = math.sqrt(max(0.0, 1.0 - (1.0 - r2) * (n_pts - 1) / (n_pts - m))) if n_pts > m else float("nan")
            out.append((name, b, r_idx, r_adj, f))
        return out

    pick = lambda t: (t[3] if t[3] == t[3] else -1.0, t[2])
    reg = fit_models(y, False)
    reg_wq = fit_models(y_wq, True)
    best, best_wq = max(reg, key=pick), max(reg_wq, key=pick)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].errorbar(lam_arr, y, yerr=[r[3] for r in rows], fmt="o", capsize=3, label="імітація (95% ДІ)")
    grid = np.linspace(lam_arr.min(), lam_arr.max(), 200)
    for name, b, r_idx, r_adj, f in reg:
        ax[0].plot(grid, f(grid) @ b, label=f"{name} (R={r_idx:.4f})")
    ax[0].set_xlabel(LABEL_LAM)
    ax[0].set_ylabel("Середній час обробки запиту W, с")
    ax[0].set_title("Вплив інтенсивності на час обробки")
    ax[0].legend(fontsize=9)
    ax[1].plot(lam_arr, [r[4] for r in rows], "s-", label="завантаження процесора 1")
    ax[1].set_xlabel(LABEL_LAM)
    ax[1].set_ylabel("Завантаження процесора")
    ax2 = ax[1].twinx()
    ax2.errorbar(lam_arr, y_wq, yerr=[r[8] for r in rows], fmt="d-", capsize=3, color="C3", label="Wq, с")
    for name, b, r_idx, r_adj, f in reg_wq:
        if "rho" in name:
            ax2.plot(grid, f(grid) @ b, "--", color="C3", lw=1, label=f"регресія Wq (R={r_idx:.4f})")
    ax2.set_ylabel("Час очікування в чергах Wq, с")
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax[1].set_title("Завантаження та час очікування в чергах")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "intensity.png"))
    plt.close(fig)

    an = {"усі рівні": anova(groups)}
    sub = [i for i, l in enumerate(lam_arr) if l <= cfg["lam_low"] + 1e-12]
    if len(sub) >= 2:
        an[f"низькі інтенсивності (lambda <= {cfg['lam_low']})"] = anova([groups[i] for i in sub])
    an_wq = {"усі рівні": anova(groups_wq)}
    return {"rows": rows, "regression": [(n, b, r, ra) for n, b, r, ra, _ in reg], "best": best[0], "anova": an,
            "regression_wq": [(n, b, r, ra) for n, b, r, ra, _ in reg_wq], "best_wq": best_wq[0], "anova_wq": an_wq}

def alternative_interpretation(out_dir, cfg, workers=None):
    rows = []
    half_T = cfg["alt_T"] / 2
    for c in (1, 2, 3):
        p = D7Params(prim_uses_cpu=True, cpu1=c)
        res = replicate(p, cfg["alt_runs"], cfg["alt_T"], 0.0, seed0=9000, workers=workers, checkpoint_step=half_T)
        first = [_area_N(r)[0] / half_T for r in res]
        second = [(_area_N(r)[1] - _area_N(r)[0]) / half_T for r in res]
        rows.append([c, 0.1 * (p.prim_time + p.p_found * p.search_mean) / c, st.fmean(r["W"] for r in res),
                     st.fmean(r["N"] for r in res), st.fmean(first), st.fmean(second),
                     st.fmean(r["X"] for r in res), st.fmean(r["rho1"] for r in res)])
    _save_csv(os.path.join(out_dir, "alt_interpretation.csv"),
              ["процесорів на комп'ютері 1", "rho1 (теор.)", "W, с", "N, шт.", "N (1-ша половина)",
               "N (2-га половина)", "пропускна здатність, 1/с", "rho1 (імітація)"], rows)
    return rows

def improvement_variants():
    b = D7Params()
    return {
        "Базовий": b,
        "А: пошук швидший, 18±2 → 12±1,33 с": replace(b, search_mean=12.0, search_half=2.0 * 12.0 / 18.0),
        "Б: канал швидший, 3 → 1 с": replace(b, channel_time=1.0),
        "В: 2 процесори на кожному комп'ютері": replace(b, cpu1=2, cpu2=2),
        "Г: 2 процесори лише на комп'ютері 1": replace(b, cpu1=2),
    }

def improvement_study(out_dir, cfg, warmup, t_run, workers=None):
    variants = improvement_variants()
    names = list(variants)
    result = {}
    keys = ("W", "Wq", "N", "rho1", "rho2")
    for lam in cfg["imp_loads"]:
        data = {}
        for name, p0 in variants.items():
            p = replace(p0, interarrival_mean=1.0 / lam, interarrival_half=0.3 / lam)
            res = replicate(p, cfg["imp_runs"], warmup + t_run, warmup, seed0=11000, workers=workers)
            data[name] = {k: [r[k] for r in res] for k in keys}
        base = data[names[0]]
        m = len(names) - 1
        rows = []
        for name in names:
            d = data[name]
            row = {"name": name}
            for key in keys:
                row[key] = ci95(d[key])
            for key in ("W", "Wq"):
                if name == names[0]:
                    row[key + "_d"] = (0.0, 0.0, float("nan"))
                else:
                    diff = st.fmean(d[key]) - st.fmean(base[key])
                    pct = 100.0 * diff / st.fmean(base[key]) if st.fmean(base[key]) > 0 else float("nan")
                    tt = stats.ttest_ind(d[key], base[key], equal_var=False)
                    row[key + "_d"] = (diff, pct, min(1.0, float(tt.pvalue) * m))
            rows.append(row)
        an = {k: anova([data[n][k] for n in names]) for k in ("W", "Wq")}
        result[lam] = {"rows": rows, "anova": an}
        _save_csv(os.path.join(out_dir, f"improvements_lambda_{lam:.2f}.csv"),
                  ["модифікація", "W, с", "+-CI95", "dW, с", "dW, %", "p(Бонферроні) W", "Wq, с", "+-CI95", "dWq, с",
                   "p(Бонферроні) Wq", "N, шт.", "rho1", "rho2"],
                  [[r["name"], r["W"][0], r["W"][1], r["W_d"][0], r["W_d"][1], r["W_d"][2],
                    r["Wq"][0], r["Wq"][1], r["Wq_d"][0], r["Wq_d"][2], r["N"][0], r["rho1"][0], r["rho2"][0]]
                   for r in rows])

    loads = cfg["imp_loads"]
    fig, ax = plt.subplots(len(loads), 2, figsize=(10, 3.9 * len(loads)), squeeze=False)
    for i, lam in enumerate(loads):
        for j, (key, lab) in enumerate((("W", "Середній час обробки W, с"), ("Wq", "Час очікування в чергах Wq, с"))):
            rows = result[lam]["rows"]
            ax[i][j].bar(range(len(rows)), [r[key][0] for r in rows], yerr=[r[key][1] for r in rows], capsize=3)
            ax[i][j].set_xticks(range(len(rows)))
            ax[i][j].set_xticklabels([r["name"].split(":")[0] if ":" in r["name"] else "Базовий" for r in rows])
            ax[i][j].set_ylabel(lab)
            ax[i][j].set_xlabel("Модифікація")
            ax[i][j].set_title(f"lambda = {lam:g} 1/с")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "improvements.png"))
    plt.close(fig)
    return result


def queue_table(out_dir, cfg, warmup, t_run, workers=None):
    res = replicate(D7Params(), cfg["queue_runs"], warmup + t_run, warmup, seed0=5000, workers=workers)
    rows = []
    for q in ("Q1", "C1", "Q2", "R"):
        lq = ci95([r["queues"][q][0] for r in res])
        wq = ci95([r["queues"][q][1] for r in res])
        rows.append([q, lq[0], lq[1], wq[0], wq[1], st.fmean(r["queues"][q][2] for r in res)])
    lq_all, wq_all = ci95([r["Lq"] for r in res]), ci95([r["Wq"] for r in res])
    rows.append(["усього", lq_all[0], lq_all[1], wq_all[0], wq_all[1], st.fmean(r["X"] for r in res)])
    _save_csv(os.path.join(out_dir, "queues_base.csv"),
              ["черга", "Lq, шт.", "+-CI95", "Wq, с", "+-CI95", "потік через чергу, 1/с"], rows)
    return rows
