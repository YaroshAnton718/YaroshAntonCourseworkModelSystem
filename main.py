from __future__ import annotations
import argparse
import math
import os
import experiments as ex
import verification as vf
from model_d7 import D7Params, analytic_response_no_queue

FULL = dict(tr_runs=5, tr_step=100.0, tr_total=200_000.0, tr_eps=0.1, tr_smooth=5, tr_delta=0.05,
            pilot_runs=30, run_eps=0.25, runs_cap=400, t_run=20_000.0, t_run_cap=100_000.0,
            dist_obs=1500, dist_tail=400.0,
            lam_levels=8, lam_levels_b=6, intens_runs=30,
            alt_runs=10, alt_T=20_000.0,
            imp_loads=(0.10, 0.45))
QUICK = dict(tr_runs=4, tr_step=100.0, tr_total=30_000.0, tr_eps=0.25, tr_smooth=5, tr_delta=0.08,
             pilot_runs=8, run_eps=0.5, runs_cap=12, t_run=5_000.0, t_run_cap=20_000.0,
             dist_obs=300, dist_tail=300.0,
             lam_levels=5, lam_levels_b=5, intens_runs=6,
             alt_runs=3, alt_T=6_000.0,
             imp_loads=(0.10, 0.45))

PRACT = 0.1

def hdr(s):
    print("\n" + f"\n{s}\n")

def anova_text(a_, yes, no):
    sig = a_["F"] > a_["F_crit"]
    s = f"{yes if sig else no}; розмах середніх між рівнями {a_['range']:.3f} с"
    if sig and a_["range"] < PRACT:
        s += f" (статистично істотний, але практично нехтовний: < {PRACT:g} с)"
    return s

def show_intensity(r):
    print(f"{'lambda, 1/с':>12}{'інтервал, с':>13}{'W, с':>10}{'+-CI95':>9}{'rho ЕОМ-1':>11}{'N, шт.':>9}{'W_max, с':>10}"
          f"{'Wq, с':>9}{'+-CI95':>9}")
    for lam, m_int, w, h, u, n, wmax, wq, hq in r["rows"]:
        print(f"{lam:>12.3f}{m_int:>13.2f}{w:>10.3f}{h:>9.3f}{u:>11.3f}{n:>9.2f}{wmax:>10.1f}{wq:>9.3f}{hq:>9.3f}")
    print("Регресійний аналіз (нормалізований фактор x = (lambda - lambda0)/delta, x в [-1; 1], формули 3.6-3.9):")
    for name, b, ri, ra in r["regression"]:
        print(f"  {name}: b = {[round(float(v), 4) for v in b]},  індекс кореляції R = {ri:.4f}, скоригований = {ra:.4f}")
    print(f"  обрана модель (за скоригованим індексом): {r['best']}")
    print("Дисперсійний аналіз (критерій Фішера, alpha = 0.05):")
    for name, a_ in r["anova"].items():
        print(f"  {name}: q={a_['q']}, n={a_['n']}, D_ф={a_['D_factor']:.4f}, D_зал={a_['D_resid']:.4f}, "
              f"F={a_['F']:.2f}, F_крит={a_['F_crit']:.2f}, p={a_['p']:.3g} -> "
              f"{anova_text(a_, 'вплив істотний', 'вплив неістотний')}")
    print("Відгук Wq (середній час очікування в чергах, с):")
    for name, b, ri, ra in r["regression_wq"]:
        print(f"  {name}: b = {[round(float(v), 4) for v in b]},  індекс кореляції R = {ri:.4f}, скоригований = {ra:.4f}")
    print(f"  обрана модель (за скоригованим індексом): {r['best_wq']}")
    for name, a_ in r["anova_wq"].items():
        print(f"  ANOVA {name}: F={a_['F']:.2f}, F_крит={a_['F_crit']:.2f}, p={a_['p']:.3g} -> "
              f"{anova_text(a_, 'вплив істотний', 'вплив неістотний')}")

def show_improvements(res):
    for lam, d in res.items():
        print(f"\n--- Навантаження lambda = {lam:g} 1/с (середній інтервал {1 / lam:.1f} с) ---")
        print(f"{'Модифікація':<62}{'W, с':>8}{'dW, %':>8}{'p(W)':>8}{'Wq, с':>8}{'dWq, с':>8}{'p(Wq)':>8}"
              f"{'N':>7}{'rho1':>6}{'rho2':>6}  прим.")
        for r in d["rows"]:
            dw, dp, pw = r["W_mean_d"]
            dq, _, pq = r["Wq_d"]
            f = lambda v: "   -" if v != v else f"{v:8.4f}"
            note = [lab + "*" for lab, dd, pp in (("W", dw, pw), ("Wq", dq, pq)) if pp == pp and pp < 0.05 and abs(dd) < PRACT]
            print(f"{r['name']:<62}{r['W_mean'][0]:>8.3f}{dp:>8.2f}{f(pw):>8}{r['Wq'][0]:>8.3f}{dq:>8.3f}{f(pq):>8}"
                  f"{r['N_mean'][0]:>7.2f}{r['util_cpu1'][0]:>6.2f}{r['util_cpu2'][0]:>6.2f}  {','.join(note)}")
        for k, lab in (("W_mean", "W"), ("Wq", "Wq")):
            a_ = d["anova"][k]
            print(f"ANOVA за фактором 'модифікація', відгук {lab}: F={a_['F']:.2f}, F_крит={a_['F_crit']:.2f}, "
                  f"p={a_['p']:.3g} -> {anova_text(a_, 'різниця істотна', 'різниця неістотна')}")
    print("p - з поправкою Бонферроні (t-критерій Велча відносно базового варіанта); p<0.05 - зміна істотна.")
    print(f"* - різниця статистично істотна, але за модулем менша за {PRACT:g} с, тобто практично нехтовна.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["verify", "transient", "stats", "intensity", "alt", "improve"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    cfg = QUICK if a.quick else FULL
    os.makedirs(a.out, exist_ok=True)
    want = lambda k: a.only in (None, k)

    p = D7Params()
    hdr("Модель D7: вхідні параметри")
    print(f"інтервал надходження: unif({p.interarrival_mean:g}, {p.interarrival_half:g}) с; "
          f"первинна обробка {p.prim_time:g} с; пошук unif({p.search_mean:g}, {p.search_half:g}) с; "
          f"канал {p.channel_time:g} с; p(знайдено на ЕОМ-1)={p.p_found}")
    print(f"аналітичний час обробки за відсутності черг: {analytic_response_no_queue(p):.2f} с")

    if want("verify"):
        hdr("Верифікація: протокол подій і таблиця верифікації")
        path, log = vf.event_log(a.out)
        print("\n".join(log[:25]), f"\n... (повний фрагмент: {path})\n")
        rows, path = vf.verification_table(a.out, n_rep=10 if a.quick else 20,
                                           t_end=10000.0 if a.quick else 40000.0, workers=a.workers)
        vf.print_table(rows)
        print(f"(таблиця збережена: {path})")

    warm, t_run, n_runs = 1000.0, cfg["t_run"], 20
    if want("transient") or want("stats") or want("intensity") or want("improve"):
        hdr("1. Перехідний період і тривалість експерименту")
        tr = ex.transient_study(a.out, cfg, a.workers)
        print(f"eps = {tr['eps']} шт.;  T за (3.1) = {tr['T_run']} с;  розкид у кінці = {tr['final_spread']:.3f};  "
              f"стале N ~ {tr['steady_N']:.3f};  перехідний період (Велч) ~ {tr['warmup']:.0f} с")
        warm = min(max(tr["warmup"] * 2, 500.0), 5000.0)
        if tr["T_run"] is not None:
            t_run = min(max(t_run, math.ceil(tr["T_run"] / 1000.0) * 1000.0), cfg["t_run_cap"])
        else:
            print(f"умова (3.1) не досягнута - беремо t_run = {t_run:.0f} с (можливо, потрібно збільшити tr_total або eps)")
        print(f"приймаємо прогрів {warm:.0f} с, тривалість збору статистики {t_run:.0f} с")

        hdr("2. Кількість повторень (нерівність Чебишева)")
        nr = ex.number_of_runs(a.out, cfg, warm, t_run, a.workers)
        print(f"sigma(середнього W за прогін) = {nr['sigma_run_mean']:.4f} с;  eps = {nr['eps']} с;  beta = {nr['beta']};  "
              f"n = sigma^2/(eps^2*(1-beta)) = {nr['n_required']}")
        n_runs = max(20, min(nr["n_required"], cfg["runs_cap"]))
        print(f"приймаємо n = {n_runs} повторень")

    if want("stats"):
        hdr("3. Статистичні характеристики часу обробки запиту")
        s = ex.statistics_and_distribution(a.out, cfg, warm, t_run, n_runs, a.workers)
        print(f"n = {s['n_runs']} прогонів; T_прогону = {s['T_run']:.0f} с; прогрів = {s['warmup']:.0f} с")
        for k, name in (("W_mean", "Середній час обробки W, с"), ("N_mean", "Запитів у системі N, шт."),
                        ("throughput", "Пропускна здатність, 1/с"), ("util_cpu1", "Завантаження ЕОМ-1"),
                        ("util_cpu2", "Завантаження ЕОМ-2")):
            print(f"{name:<34}{s[k][0]:.4f} +- {s[k][1]:.4f}")
        print(f"{'СКВ часу обробки (за запитами), с':<34}{s['W_std']:.3f}")
        print(f"{'min / max часу обробки, с':<34}{s['W_min']:.3f} / {s['W_max']:.3f}")
        print(f"\nІдентифікація закону розподілу, вибірка з {s['sample_n']} незалежних спостережень: "
              f"m={s['sample_mean']:.2f}, s={s['sample_std']:.2f}, min={s['sample_min']:.2f}, max={s['sample_max']:.2f}")
        print(f"{'закон':<34}{'параметри':<58}{'chi2':>9}{'df':>5}{'p':>9}")
        for name, pars, chi2, df, pv in s["dist_table"]:
            print(f"{name:<34}{pars:<58}{chi2:>9.2f}{df:>5}{pv:>9.4f}")
        print(f"Прийнятий закон: {s['best_dist']} ({s['best_rule']}; гіпотеза не відхиляється, якщо p > 0.05)")

    if want("intensity"):
        warm_i, t_i = max(warm, 5000.0), t_run
        hdr("4а. Вплив інтенсивності (базова інтерпретація: процесор зайнятий лише первинною обробкою)")
        r = ex.intensity_study(a.out, "base", D7Params(), 0.05, 0.48, cfg["lam_levels"], 0.15,
                               warm_i, t_i, n_runs, a.workers)
        show_intensity(r)
        hdr("4б. Вплив інтенсивності (процесор зайнятий і пошуком, 3 процесори на ЕОМ)")
        r = ex.intensity_study(a.out, "cpu_hold", D7Params(search_occupies_cpu=True, cpu_channels=3),
                               0.05, 0.14, cfg["lam_levels_b"], 0.08, warm_i, 2 * t_i, n_runs, a.workers)
        show_intensity(r)

    if want("alt"):
        hdr("5. Альтернативна інтерпретація: процесор зайнятий і під час пошуку")
        rows = ex.alternative_interpretation(a.out, cfg, a.workers)
        print(f"{'процесорів':>11}{'rho ЕОМ-1':>11}{'W, с':>10}{'N':>9}{'N 1-ша пол.':>13}{'N 2-га пол.':>13}{'lambda_вих, 1/с':>17}")
        for c, rho, w, n, n1, n2, th in rows:
            print(f"{c:>11}{rho:>11.2f}{w:>10.1f}{n:>9.1f}{n1:>13.1f}{n2:>13.1f}{th:>17.4f}")
        print("Якщо N у другій половині помітно більше, ніж у першій - черга зростає необмежено (режим нестабільний).")

    if want("improve"):
        hdr("6. Пропозиції щодо поліпшення: порівняння модифікацій системи")
        res = ex.improvement_study(a.out, cfg["imp_loads"], max(warm, 5000.0), t_run, n_runs, a.workers)
        show_improvements(res)

if __name__ == "__main__":
    main()