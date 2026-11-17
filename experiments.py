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
from verification import ci95

plt.rcParams["figure.dpi"] = 110

def _save_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def transient_study(out_dir, cfg, workers=None):
    p = D7Params()
    n_runs, step, t_total, eps = cfg["tr_runs"], cfg["tr_step"], cfg["tr_total"], cfg["tr_eps"]
    res = replicate(p, n_runs, t_total, 0.0, seed0=1, workers=workers, checkpoint_step=step)
    ts = np.array([c[0] for c in res[0]["checkpoints"]])
    areas = np.array([[c[1] for c in r["checkpoints"]] for r in res])
    ybar = areas / ts
    spread = ybar.max(axis=0) - ybar.min(axis=0)
    ok = spread < eps
    T_run = None
    for k in range(len(ts)):
        if ok[k:].all():
            T_run = float(ts[k])
            break

    win = np.diff(np.vstack([np.zeros((1, n_runs)), areas.T]), axis=0).T / step
    avg = win.mean(axis=0)
    w = cfg["tr_smooth"]
    smooth = np.convolve(avg, np.ones(w) / w, mode="same")
    steady = avg[len(avg) // 2:].mean()
    noise = smooth[len(avg) // 2: -(w // 2)].std()
    tol = max(cfg["tr_delta"] * steady, 3.0 * noise)
    head = max(len(ts) // 4, w)
    inside = np.abs(smooth[:head] - steady) <= tol
    inside[: w // 2] = False
    hit = np.flatnonzero(inside)
    warm = float(ts[hit[0]] - step) if hit.size else float(ts[head - 1])
    warm = max(warm, 0.0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for i in range(n_runs):
        ax[0].plot(ts, ybar[i], lw=0.9, label=f"прогін {i + 1}")
    ax[0].axvline(T_run if T_run else t_total, color="k", ls="--", label="T")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Модельний час t, с")
    ax[0].set_ylabel("Середня кількість запитів у системі y(t), шт.")
    ax[0].set_title("Збіжність середнього (формули 3.1-3.2)")
    ax[0].legend(fontsize=8)
    ax[1].plot(ts, avg, color="0.7", label="середнє за прогонами, вікно %d с" % step)
    ax[1].plot(ts, smooth, "r", label="ковзне середнє (%d вікон)" % w)
    ax[1].axhline(steady, color="g", ls=":", label="стале значення")
    ax[1].axvline(warm, color="k", ls="--", label=f"прогрів ~ {warm:.0f} с")
    ax[1].set_xlim(0, min(t_total, max(20 * warm, 5000)))
    ax[1].set_xlabel("Модельний час t, с")
    ax[1].set_ylabel("Кількість запитів у системі N, шт.")
    ax[1].set_title("Перехідний період")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "transient.png"))
    plt.close(fig)

    info = {"eps": eps, "T_run": T_run, "warmup": warm, "steady_N": float(steady),
            "final_spread": float(spread[-1])}
    _save_csv(os.path.join(out_dir, "transient.csv"), ["t, с"] + [f"y{i + 1}" for i in range(n_runs)] + ["spread"],
              [[ts[k]] + list(ybar[:, k]) + [spread[k]] for k in range(len(ts))])
    return info

def number_of_runs(out_dir, cfg, warmup, t_run, workers=None):
    pilot = replicate(D7Params(), cfg["pilot_runs"], warmup + t_run, warmup, seed0=100, workers=workers)
    vals = [r["W_mean"] for r in pilot]
    sigma = st.stdev(vals)
    eps, beta = cfg["run_eps"], 0.95
    n = math.ceil(sigma ** 2 / (eps ** 2 * (1 - beta)))
    return {"sigma_run_mean": sigma, "eps": eps, "beta": beta, "n_required": n,
            "pilot_mean": st.fmean(vals)}

def fit_gmm2(x, iters=500):
    x = np.asarray(x)
    mu = np.array([np.percentile(x, 25), np.percentile(x, 75)])
    sd = np.array([x.std() / 2, x.std() / 2])
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        pdf = np.stack([w[k] * stats.norm.pdf(x, mu[k], sd[k]) for k in range(2)])
        resp = pdf / pdf.sum(axis=0)
        nk = resp.sum(axis=1)
        w = nk / len(x)
        mu = (resp * x).sum(axis=1) / nk
        sd = np.sqrt((resp * (x - mu[:, None]) ** 2).sum(axis=1) / nk)
    return w, mu, sd

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
    return chi2, df, pval

def statistics_and_distribution(out_dir, cfg, warmup, t_run, n_runs, workers=None):
    p = D7Params()
    res = replicate(p, n_runs, warmup + t_run, warmup, seed0=2000, workers=workers)
    out = {"n_runs": n_runs, "T_run": t_run, "warmup": warmup}
    for key in ("W_mean", "N_mean", "throughput", "util_cpu1", "util_cpu2"):
        out[key] = ci95([r[key] for r in res])
    out["W_std"] = st.fmean(r["W_std"] for r in res)
    out["W_min"] = min(r["W_min"] for r in res)
    out["W_max"] = max(r["W_max"] for r in res)

    runs = replicate(p, cfg["dist_obs"], warmup + cfg["dist_tail"], warmup, seed0=50000, workers=workers,
                     keep_records=True)
    sample = []
    for r in runs:
        births = [(b, w) for b, _, w in r["records"] if b >= warmup]
        sample.append(min(births)[1])
    x = np.array(sample)
    out["sample_n"] = len(x)
    out["sample_mean"], out["sample_std"] = float(x.mean()), float(x.std(ddof=1))
    out["sample_min"], out["sample_max"] = float(x.min()), float(x.max())

    cands = {}
    mu, sd = stats.norm.fit(x)
    cands["нормальний"] = (lambda v: stats.norm.cdf(v, mu, sd), 2, f"m={mu:.2f}, s={sd:.2f}",
                           lambda v: stats.norm.pdf(v, mu, sd))
    s, loc, sc = stats.lognorm.fit(x, floc=0)
    cands["логнормальний"] = (lambda v: stats.lognorm.cdf(v, s, loc, sc), 2, f"s={s:.3f}, scale={sc:.2f}",
                              lambda v: stats.lognorm.pdf(v, s, loc, sc))
    a, loc_g, sc_g = stats.gamma.fit(x, floc=0)
    cands["Ерланга/гамма"] = (lambda v: stats.gamma.cdf(v, a, loc_g, sc_g), 2, f"k={a:.2f}, scale={sc_g:.2f}",
                              lambda v: stats.gamma.pdf(v, a, loc_g, sc_g))
    ww, mm, ss = fit_gmm2(x)
    mix_cdf = lambda v: sum(ww[k] * stats.norm.cdf(v, mm[k], ss[k]) for k in range(2))
    mix_pdf = lambda v: sum(ww[k] * stats.norm.pdf(v, mm[k], ss[k]) for k in range(2))
    cands["суміш двох нормальних"] = (
        mix_cdf, 5, f"w=({ww[0]:.2f}, {ww[1]:.2f}), m=({mm[0]:.1f}, {mm[1]:.1f}), s=({ss[0]:.1f}, {ss[1]:.1f})", mix_pdf)

    w1 = p.p_found
    a1 = p.prim_time + p.search_mean - p.search_half
    b1 = p.prim_time + p.search_mean + p.search_half
    n_ch = 2 if p.return_result else 1
    a2 = 2 * (p.prim_time + p.search_mean - p.search_half) + n_ch * p.channel_time
    b2 = 2 * (p.prim_time + p.search_mean + p.search_half) + n_ch * p.channel_time
    st_cdf = lambda v: w1 * stats.uniform.cdf(v, a1, b1 - a1) + (1 - w1) * stats.triang.cdf(v, 0.5, a2, b2 - a2)
    st_pdf = lambda v: w1 * stats.uniform.pdf(v, a1, b1 - a1) + (1 - w1) * stats.triang.pdf(v, 0.5, a2, b2 - a2)
    cands["суміш рівномірного і трикутного"] = (
        st_cdf, 0, f"w={w1:.2f}, U({a1:.1f}; {b1:.1f}), Tri({a2:.1f}; {b2:.1f}) - з параметрів моделі", st_pdf)

    table = []
    for name, (cdf, k, pars, _) in cands.items():
        chi2, df, pv = chi2_test(x, cdf, k)
        table.append((name, pars, chi2, df, pv))
    out["dist_table"] = table
    best = max((t for t in table if not math.isnan(t[4])), key=lambda t: t[4], default=None)
    theory = "суміш рівномірного і трикутного"
    th_p = next(t[4] for t in table if t[0] == theory)
    if th_p > 0.05:
        out["best_dist"], out["best_rule"] = theory, "гіпотеза з логіки моделі не відхилена (p > 0.05)"
    else:
        out["best_dist"], out["best_rule"] = (best[0] if best else None), "найбільше p-value"

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(x, bins=int(round(2 * len(x) ** 0.4)), density=True, alpha=0.55, label="гістограма")
    grid = np.linspace(x.min(), x.max(), 400)
    for name, (_, _, _, pdf) in cands.items():
        ax.plot(grid, pdf(grid), lw=1.6, label=name)
    ax.set_xlabel("Час обробки запиту, с")
    ax.set_ylabel("Щільність, 1/с")
    ax.set_title(f"Розподіл часу обробки запиту (вибірка {len(x)} незалежних спостережень)")
    ax.legend(fontsize=8)
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
    if d_r == 0.0:
        F = 0.0 if d_f == 0.0 else float("inf")
    else:
        F = d_f / d_r
    means = g.mean(axis=1)
    return {"q": q, "n": n, "D_factor": d_f, "D_resid": d_r, "F": F,
            "range": float(means.max() - means.min()),
            "F_crit": stats.f.ppf(0.95, q - 1, q * (n - 1)),
            "p": 1.0 if F == 0.0 else float(1 - stats.f.cdf(F, q - 1, q * (n - 1)))}

def intensity_study(out_dir, tag, base, lam_min, lam_max, levels, lam_low,
                    warmup, t_run, n_runs, workers=None):
    lams = np.linspace(lam_min, lam_max, levels)
    rows, groups, groups_wq = [], [], []
    for lam in lams:
        p = replace(base, interarrival_mean=1.0 / lam, interarrival_half=0.3 / lam)
        res = replicate(p, n_runs, warmup + t_run, warmup, seed0=7000, workers=workers)
        w = [r["W_mean"] for r in res]
        wq = [r["Wq"] for r in res]
        m, h = ci95(w)
        mq, hq = ci95(wq)
        groups.append(w)
        groups_wq.append(wq)
        rows.append([lam, 1.0 / lam, m, h, st.fmean(r["util_cpu1"] for r in res),
                     st.fmean(r["N_mean"] for r in res), max(r["W_max"] for r in res), mq, hq])
    _save_csv(os.path.join(out_dir, f"intensity_{tag}.csv"),
              ["lambda, 1/с", "середній інтервал, с", "W, с", "+-CI95, с", "завантаження ЕОМ-1",
               "N, шт.", "W_max, с", "Wq (очікування в чергах), с", "+-CI95 Wq, с"], rows)
    lam_arr = np.array([r[0] for r in rows])
    y = np.array([r[2] for r in rows])
    y_wq = np.array([r[7] for r in rows])

    svc = base.prim_time + (base.search_mean if base.search_occupies_cpu else 0.0)
    c, pf = base.cpu_channels, base.p_found
    k_c = math.sqrt(2 * (c + 1)) - 1

    def poly(l):
        x = normalize(l, lam_arr.min(), lam_arr.max())
        return np.column_stack([np.ones_like(x), x, x ** 2, x ** 3])

    def stations(l):
        r1, r2 = svc * l / c, (1 - pf) * svc * l / c
        cols = [r1 ** k_c / (1 - r1), r2 ** k_c / (1 - r2)]
        if base.channel_single:
            rk = (1 - pf) * base.channel_time * l
            cols.append(rk / (1 - rk))
        return np.column_stack(cols)

    queue_w = lambda l: np.column_stack([np.ones_like(l), stations(l)])
    queue_q = stations
    st_name = "b1*f(rho1) + b2*f(rho2)" + (" + b3*f(rho_кан)" if base.channel_single else "")

    def solve(F, resp, nonneg, free_intercept):
        if not nonneg:
            return least_squares(F, resp)
        if free_intercept:
            A = np.column_stack([F[:, 0], -F[:, 0], F[:, 1:]])
            b = nnls(A, resp)[0]
            return np.concatenate([[b[0] - b[1]], b[2:]])
        return nnls(F, resp)[0]

    def fit_models(resp, for_wq):
        specs = [("поліном 3-го степеня від x", poly, False),
                 ((st_name if for_wq else "b0 + " + st_name) + ", bi >= 0", queue_q if for_wq else queue_w, True)]
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
    best = max(reg, key=pick)
    best_wq = max(reg_wq, key=pick)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    ax[0].errorbar(lam_arr, y, yerr=[r[3] for r in rows], fmt="o", capsize=3, label="імітація (95% ДІ)")
    grid = np.linspace(lam_arr.min(), lam_arr.max(), 200)
    for name, b, r_idx, r_adj, f in reg:
        ax[0].plot(grid, f(grid) @ b, label=f"{name} (R={r_idx:.4f})")
    ax[0].set_xlabel("Інтенсивність надходження lambda, 1/с")
    ax[0].set_ylabel("Середній час обробки запиту W, с")
    ax[0].set_title(f"Вплив інтенсивності на час обробки ({tag})")
    ax[0].legend(fontsize=8)
    ax[1].plot(lam_arr, [r[4] for r in rows], "s-", label="завантаження ЕОМ-1")
    ax[1].errorbar(lam_arr, y_wq, yerr=[r[8] for r in rows], fmt="d-", capsize=3, color="C3",
                   label="Wq, с (очікування в чергах)")
    for name, b, r_idx, r_adj, f in reg_wq:
        if "rho" in name:
            ax[1].plot(grid, f(grid) @ b, "--", color="C3", lw=1, label=f"регресія Wq (R={r_idx:.4f})")
    ax[1].set_xlabel("Інтенсивність надходження lambda, 1/с")
    ax[1].set_ylabel("Середній час очікування в чергах Wq, с")
    ax[1].set_title("Відгук Wq")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"intensity_{tag}.png"))
    plt.close(fig)

    an = {"усі рівні": anova(groups)}
    sub = [i for i, l in enumerate(lam_arr) if l <= lam_low + 1e-12]
    if len(sub) >= 2:
        an[f"низькі інтенсивності (lambda <= {lam_low})"] = anova([groups[i] for i in sub])
    an_wq = {"усі рівні": anova(groups_wq)}
    if len(sub) >= 2:
        an_wq[f"низькі інтенсивності (lambda <= {lam_low})"] = anova([groups_wq[i] for i in sub])
    return {"rows": rows, "regression": [(n, b, r, ra) for n, b, r, ra, _ in reg], "best": best[0], "anova": an,
            "regression_wq": [(n, b, r, ra) for n, b, r, ra, _ in reg_wq], "best_wq": best_wq[0], "anova_wq": an_wq}

def alternative_interpretation(out_dir, cfg, workers=None):
    rows = []
    half_T = cfg["alt_T"] / 2
    for c in (1, 2, 3, 4):
        p = D7Params(search_occupies_cpu=True, cpu_channels=c)
        res = replicate(p, cfg["alt_runs"], cfg["alt_T"], 0.0, seed0=9000, workers=workers,
                        checkpoint_step=half_T)
        first = [r["checkpoints"][0][1] / half_T for r in res]
        second = [(r["checkpoints"][1][1] - r["checkpoints"][0][1]) / half_T for r in res]
        rows.append([c, 20.0 / (10.0 * c), st.fmean(r["W_mean"] for r in res), st.fmean(r["N_mean"] for r in res),
                     st.fmean(first), st.fmean(second), st.fmean(r["throughput"] for r in res)])
    _save_csv(os.path.join(out_dir, "alt_interpretation.csv"),
              ["процесорів на ЕОМ", "rho ЕОМ-1", "W, с", "N, шт.", "N (1-ша половина)", "N (2-га половина)",
               "пропускна здатність, 1/с"], rows)
    return rows

def improvement_variants():
    b = D7Params()
    return {
        "Базовий": b,
        "A: p_found 0.5 -> 0.7 (більша частка знайденого на ЕОМ-1)": replace(b, p_found=0.7),
        "B: пошук швидший, 18+-2 -> 12+-1.33 с": replace(b, search_mean=12.0, search_half=2.0 * 12.0 / 18.0),
        "C: канал швидший, 3 -> 1 с": replace(b, channel_time=1.0),
        "D: 2 процесори на кожній ЕОМ": replace(b, cpu_channels=2),
    }

def improvement_study(out_dir, loads, warmup, t_run, n_runs, workers=None):
    variants = improvement_variants()
    names = list(variants)
    result = {}
    for lam in loads:
        data = {}
        for name, p0 in variants.items():
            p = replace(p0, interarrival_mean=1.0 / lam, interarrival_half=0.3 / lam)
            res = replicate(p, n_runs, warmup + t_run, warmup, seed0=11000, workers=workers)
            data[name] = {k: [r[k] for r in res] for k in ("W_mean", "Wq", "N_mean", "util_cpu1", "util_cpu2")}
        base = data[names[0]]
        m = len(names) - 1
        rows = []
        for name in names:
            d = data[name]
            row = {"name": name}
            for key in ("W_mean", "Wq", "N_mean", "util_cpu1", "util_cpu2"):
                row[key] = ci95(d[key])
            for key in ("W_mean", "Wq"):
                if name == names[0]:
                    row[key + "_d"] = (0.0, 0.0, float("nan"))
                else:
                    diff = st.fmean(d[key]) - st.fmean(base[key])
                    pct = 100.0 * diff / st.fmean(base[key]) if st.fmean(base[key]) > 0 else float("nan")
                    tt = stats.ttest_ind(d[key], base[key], equal_var=False)
                    row[key + "_d"] = (diff, pct, min(1.0, float(tt.pvalue) * m))
            rows.append(row)
        an = {k: anova([data[n][k] for n in names]) for k in ("W_mean", "Wq")}
        result[lam] = {"rows": rows, "anova": an}
        _save_csv(os.path.join(out_dir, f"improvements_lambda_{lam:.2f}.csv"),
                  ["модифікація", "W, с", "+-CI95", "dW, с", "dW, %", "p(Бонферроні) W", "Wq, с", "+-CI95", "dWq, с",
                   "p(Бонферроні) Wq", "N, шт.", "завантаження ЕОМ-1", "завантаження ЕОМ-2"],
                  [[r["name"], r["W_mean"][0], r["W_mean"][1], r["W_mean_d"][0], r["W_mean_d"][1], r["W_mean_d"][2],
                    r["Wq"][0], r["Wq"][1], r["Wq_d"][0], r["Wq_d"][2], r["N_mean"][0],
                    r["util_cpu1"][0], r["util_cpu2"][0]] for r in rows])

    fig, ax = plt.subplots(1, len(loads) * 2, figsize=(5 * len(loads) * 2, 4.4), squeeze=False)
    k = 0
    for lam in loads:
        for key, lab in (("W_mean", "Середній час обробки W, с"), ("Wq", "Час очікування в чергах Wq, с")):
            rows = result[lam]["rows"]
            ax[0][k].bar(range(len(rows)), [r[key][0] for r in rows], yerr=[r[key][1] for r in rows], capsize=3)
            ax[0][k].set_xticks(range(len(rows)))
            ax[0][k].set_xticklabels([r["name"].split(":")[0] if ":" in r["name"] else "База" for r in rows])
            ax[0][k].set_ylabel(lab)
            ax[0][k].set_xlabel("Модифікація")
            ax[0][k].set_title(f"lambda = {lam:g} 1/с")
            k += 1
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "improvements.png"))
    plt.close(fig)
    return result