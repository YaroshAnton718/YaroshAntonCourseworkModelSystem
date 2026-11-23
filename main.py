from __future__ import annotations
import argparse
import math
import os
import experiments as ex
import verification as vf
from model_d7 import D7Params, analytic_response_no_queue

FULL = dict(tr_runs=5, tr_step=1000.0, tr_total=1_000_000.0, tr_eps=0.5,
            wl_runs=300, wl_step=100.0, wl_total=30_000.0, wl_smooth=10, wl_delta=0.05,
            pilot_runs=30, run_eps=1.0, t_run_cap=700_000.0, dist_runs=150,
            lam_lo=0.03, lam_hi=0.10, lam_levels=8, lam_low=0.05, intens_runs=30,
            alt_runs=10, alt_T=50_000.0,
            imp_loads=(0.06, 0.10), imp_runs=30, queue_runs=30)
QUICK = dict(tr_runs=4, tr_step=1000.0, tr_total=100_000.0, tr_eps=1.5,
             wl_runs=40, wl_step=100.0, wl_total=20_000.0, wl_smooth=10, wl_delta=0.08,
             pilot_runs=8, run_eps=2.0, t_run_cap=60_000.0, dist_runs=40,
             lam_lo=0.03, lam_hi=0.10, lam_levels=5, lam_low=0.05, intens_runs=6,
             alt_runs=3, alt_T=10_000.0,
             imp_loads=(0.06, 0.10), imp_runs=6, queue_runs=6)

PRACT = 0.1  

def hdr(s):
    print("\n" + f"\n{s}\n")

def anova_text(a_, yes, no):
    sig = a_["F"] > a_["F_crit"]
    s = f"{yes if sig else no}; розмах середніх між рівнями {a_['range']:.3f}"
    if sig and a_["range"] < PRACT:
        s += f" (статистично істотний, але практично нехтовний: < {PRACT:g})"
    return s

def show_intensity(r):
    print(f"{'lambda, 1/с':>12}{'інтервал, с':>13}{'W, с':>10}{'+-CI95':>9}{'rho1':>8}{'rho2':>8}{'N, шт.':>9}"
          f"{'Wq, с':>9}{'+-CI95':>9}")
    for lam, m_int, w, h, u1, u2, n, wq, hq in r["rows"]:
        print(f"{lam:>12.3f}{m_int:>13.2f}{w:>10.3f}{h:>9.3f}{u1:>8.3f}{u2:>8.3f}{n:>9.2f}{wq:>9.3f}{hq:>9.3f}")
    print("Регресійний аналіз (нормалізований фактор x в [-1; 1], формули 4.5-4.6):")
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
        print(f"  ANOVA {name}: F={a_['F']:.2f}, F_крит={a_['F_crit']:.2f}, p={a_['p']:.3g}")

def show_improvements(res):
    for lam, d in res.items():
        print(f"\n--- Навантаження lambda = {lam:g} 1/с (середній інтервал {1 / lam:.1f} с) ---")
        print(f"{'Модифікація':<44}{'W, с':>8}{'+-':>7}{'dW, %':>8}{'p(W)':>8}{'Wq, с':>8}{'dWq, с':>8}{'p(Wq)':>8}"
              f"{'N':>7}{'rho1':>6}{'rho2':>6}")
        for r in d["rows"]:
            dw, dp, pw = r["W_d"]
            dq, _, pq = r["Wq_d"]
            f = lambda v: "   -" if v != v else f"{v:8.4f}"
            print(f"{r['name']:<44}{r['W'][0]:>8.3f}{r['W'][1]:>7.3f}{dp:>8.2f}{f(pw):>8}{r['Wq'][0]:>8.3f}{dq:>8.3f}"
                  f"{f(pq):>8}{r['N'][0]:>7.2f}{r['rho1'][0]:>6.2f}{r['rho2'][0]:>6.2f}")
        for k in ("W", "Wq"):
            a_ = d["anova"][k]
            print(f"ANOVA за фактором 'модифікація', відгук {k}: F={a_['F']:.2f}, F_крит={a_['F_crit']:.2f}, "
                  f"p={a_['p']:.3g} -> {anova_text(a_, 'різниця істотна', 'різниця неістотна')}")
    print("p - з поправкою Бонферроні (t-критерій Велча відносно базового варіанта); p<0.05 - зміна істотна.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["verify", "transient", "stats", "intensity", "alt", "improve", "queues"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--warm", type=float, default=None)
    ap.add_argument("--trun", type=float, default=None)
    a = ap.parse_args()
    cfg = QUICK if a.quick else FULL
    os.makedirs(a.out, exist_ok=True)
    want = lambda k: a.only in (None, k)

    p = D7Params()
    hdr("Модель D7: вхідні параметри")
    print(f"інтервал надходження: unif({p.interarrival_mean:g}, {p.interarrival_half:g}) с; "
          f"первинна обробка {p.prim_time:g} с; пошук unif({p.search_mean:g}, {p.search_half:g}) с; "
          f"канал {p.channel_time:g} с; p(знайдено на комп'ютері 1)={p.p_found}")
    print(f"аналітичний час обробки за відсутності черг: {analytic_response_no_queue(p):.2f} с")
    lam = 1.0 / p.interarrival_mean
    print(f"завантаження: rho1 = rho2 = lambda*p*t_пошуку = {lam * p.p_found * p.search_mean:.2f}; "
          f"межа стабільності lambda = {1.0 / (p.p_found * p.search_mean):.4f} 1/с")

    if want("verify"):
        hdr("Верифікація: протокол подій і таблиця верифікації")
        first, second = vf.event_log(os.path.join(a.out, "event_log.txt"))
        print("\n".join(first[:25]), "\n... (повний протокол: event_log.txt)\n")
        rows = vf.run_verification(workers=a.workers)
        vf.print_verification(rows)
        ex._save_csv(os.path.join(a.out, "verification_table.csv"),
                     ["N", "опис", "T, с", "n", "змінна", "очікуване", "отримане", "допуск/CI99", "висновок"], rows)

    warm, t_run, n_runs = 1000.0, cfg["t_run_cap"], 20
    if a.warm is not None and a.trun is not None:
        warm, t_run = a.warm, a.trun
    elif a.only in (None, "transient", "stats", "intensity", "improve"):
        hdr("1. Перехідний період і тривалість експерименту")
        tr = ex.transient_study(a.out, cfg, a.workers)
        print(f"eps = {tr['eps']} шт.;  T за (4.1)-(4.2) = {tr['T_run']} с;  розкид у кінці = {tr['final_spread']:.3f};  "
              f"стале N ~ {tr['steady_N']:.3f};  перехідний період (Велч) ~ {tr['welch']:.0f} с")
        warm = max(500.0, math.ceil(2 * tr["welch"] / 500.0) * 500.0)
        if tr["T_run"] is not None:
            t_run = min(math.ceil(tr["T_run"] / 1000.0) * 1000.0, cfg["t_run_cap"])
        else:
            print(f"умова (4.2) не досягнута - беремо t_run = {t_run:.0f} с")
        print(f"приймаємо прогрів {warm:.0f} с, тривалість збору статистики {t_run:.0f} с")

        hdr("2. Кількість повторень (нерівність Чебишева)")
        nr = ex.number_of_runs(cfg, warm, t_run, a.workers)
        print(f"sigma(W за прогін) = {nr['sigma_run']:.4f} с;  eps = {nr['eps']} с;  beta = {nr['beta']};  "
              f"n = sigma^2/(eps^2*(1-beta)) = {nr['n_required']}")
        n_runs = max(20, nr["n_required"])
        print(f"приймаємо n = {n_runs} повторень")

    if want("stats"):
        hdr("3. Статистичні характеристики часу обробки запиту")
        s = ex.statistics_and_distribution(a.out, cfg, warm, t_run, n_runs, a.workers)
        print(f"n = {s['n_runs']} прогонів; T_прогону = {s['T_run']:.0f} с; прогрів = {s['warmup']:.0f} с")
        print(f"{'Середній час обробки W, с':<34}{s['W_mean']:.4f} +- {s['W_ci']:.4f}")
        print(f"{'СКВ W між прогонами, с':<34}{s['W_std']:.4f}")
        print(f"{'min / max W між прогонами, с':<34}{s['W_min']:.3f} / {s['W_max']:.3f}")
        for k, name in (("N", "Запитів у системі N, шт."), ("X", "Пропускна здатність X, 1/с"),
                        ("rho1", "Завантаження процесора 1"), ("rho2", "Завантаження процесора 2"),
                        ("Lq", "Середня довжина черг Lq, шт."), ("Wq", "Час очікування в чергах Wq, с")):
            print(f"{name:<34}{s[k][0]:.4f} +- {s[k][1]:.4f}")
        print(f"інваріанти в усіх прогонах: {s['inv']}; асиметрія {s['skew']:.3f}, ексцес {s['kurt']:.3f}")
        print(f"\nІдентифікація закону розподілу W (вибірка {s['n_runs']} прогонів, інтервалів гістограми {s['bins']} після об'єднання)")
        print(f"{'закон':<24}{'параметри':<34}{'chi2':>9}{'df':>5}{'p':>9}")
        for name, pars, chi2, df, pv in s["dist_table"]:
            print(f"{name:<24}{pars:<34}{chi2:>9.2f}{df:>5}{pv:>9.4f}")
        print(f"Прийнятий закон: {s['best_dist']} (гіпотеза не відхиляється, якщо p > 0.05)")

    if a.only == "queues" or a.only is None:
        hdr("3б. Черги за позиціями в базовому режимі")
        qw = a.warm if a.warm is not None else warm
        qt = a.trun if a.trun is not None else t_run
        rows = ex.queue_table(a.out, cfg, qw, qt, a.workers)
        print(f"{'черга':<8}{'Lq, шт.':>10}{'+-':>8}{'Wq, с':>10}{'+-':>8}{'потік, 1/с':>12}")
        for q, lq, hl, wq, hw, fl in rows:
            print(f"{q:<8}{lq:>10.4f}{hl:>8.4f}{wq:>10.3f}{hw:>8.3f}{fl:>12.4f}")

    if want("intensity"):
        hdr("4. Вплив інтенсивності надходження запитів на час обробки")
        r = ex.intensity_study(a.out, cfg, warm, t_run, a.workers)
        show_intensity(r)

    if want("alt"):
        hdr("5. Альтернатива: первинна обробка також займає процесор")
        rows = ex.alternative_interpretation(a.out, cfg, a.workers)
        print(f"{'процесорів':>11}{'rho1 теор.':>12}{'W, с':>10}{'N':>9}{'N 1-ша пол.':>13}{'N 2-га пол.':>13}"
              f"{'X, 1/с':>9}{'rho1 імітац.':>14}")
        for c, rho, w, n, n1, n2, th, ru in rows:
            print(f"{c:>11}{rho:>12.2f}{w:>10.1f}{n:>9.1f}{n1:>13.1f}{n2:>13.1f}{th:>9.4f}{ru:>14.3f}")
        print("Якщо N у другій половині помітно більше, ніж у першій - черга зростає необмежено (режим нестабільний).")

    if want("improve"):
        hdr("6. Пропозиції щодо поліпшення: порівняння модифікацій системи")
        res = ex.improvement_study(a.out, cfg, warm, t_run, a.workers)
        show_improvements(res)

if __name__ == "__main__":
    main()
