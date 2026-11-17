from __future__ import annotations
import csv
import math
import os
import statistics as st
from scipy import stats
from model_d7 import D7Params, analytic_response_no_queue, replicate, simulate
from model_d7 import replace

def ci(values, level=0.95):
    n = len(values)
    m = st.fmean(values)
    if n < 2:
        return m, 0.0
    h = stats.t.ppf(0.5 + level / 2, n - 1) * st.stdev(values) / math.sqrt(n)
    return m, h

def ci95(values):
    return ci(values, 0.95)

def ci99(values):
    return ci(values, 0.99)

def event_log(out_dir, n_lines=60):
    r = simulate(D7Params(), t_end=120.0, seed=7, log_limit=n_lines)
    chan = replace(D7Params(), interarrival_mean=2.5, interarrival_half=0.0, p_found=0.0, search_half=0.0)
    r2 = simulate(chan, t_end=60.0, seed=7, log_limit=400)
    frag2 = [ln for ln in r2["log"] if float(ln[2:14]) >= 42.0][:30]
    path = os.path.join(out_dir, "event_log.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Фрагмент 1. Протокол подій (D7, параметри за замовчуванням, зерно ГВЧ = 7)\n")
        f.write("вхід  - запуск переходу (маркери вилучено з вхідних позицій)\n")
        f.write("вихід - завершення затримки (маркери додано у вихідні позиції)\n")
        f.write("M - позиції з ненульовим маркіруванням (CPU1, CPU2, CH12, CH21 - вільні ресурси; "
                "відсутність позиції в M означає, що ресурс зайнятий)\n\n")
        f.write("\n".join(r["log"]) + "\n\n")
        f.write("Фрагмент 2. Черга до каналу (p=0, інтервал надходження 2.5 с, передача 3 с, детерміновані затримки)\n")
        f.write("Канал ЕОМ-1 -> ЕОМ-2 передає по одному повідомленню: поки CH12 відсутня в M, "
                "наступні повідомлення чекають у C1\n\n")
        f.write("\n".join(frag2) + "\n")
    return path, r["log"]

def _row(name, expected, observed, half, tol_abs=None, note=""):
    if tol_abs is not None:
        ok = abs(observed - expected) <= tol_abs
    else:
        ok = abs(observed - expected) <= half
    return {"scenario": name, "expected": expected, "observed": observed,
            "ci99_half": half, "ok": ok, "note": note}

def verification_table(out_dir, n_rep=20, t_end=40000.0, warmup=500.0, workers=None):
    base = D7Params()
    rows = []
    inv_ok = True

    def series(p, key, **kw):
        res = replicate(p, n_rep, kw.get("t_end", t_end), kw.get("warmup", warmup), workers=workers)
        nonlocal inv_ok
        inv_ok &= all(a == b for r in res for _, a, b in r["invariants"])
        return res, ci99([r[key] for r in res])

    low = replace(base, interarrival_mean=300.0, interarrival_half=0.0)

    for name, p in (("V1 p=0.5, мала інтенсивність (1/300 с)", low),
                    ("V2 p=1 (усе знайдено на ЕОМ-1)", replace(low, p_found=1.0)),
                    ("V3 p=0 (усе йде на ЕОМ-2)", replace(low, p_found=0.0)),
                    ("V4 p=0.5, без зворотної передачі результату", replace(low, return_result=False))):
        res, (m, h) = series(p, "W_mean", t_end=3_000_000.0, warmup=2000.0)
        rows.append(_row(name + ": середній час обробки, с", analytic_response_no_queue(p), m, h,
                         note="аналітично (сума затримок)"))

    det = replace(low, search_half=0.0)
    for name, p, exp in (("V5 детерміновані затримки, p=1: W_min=W_max", replace(det, p_found=1.0), 20.0),
                         ("V6 детерміновані затримки, p=0: W_min=W_max", replace(det, p_found=0.0), 46.0)):
        r = simulate(p, 20000.0, 0.0, seed=1)
        ok = abs(r["W_min"] - exp) < 1e-9 and abs(r["W_max"] - exp) < 1e-9
        rows.append({"scenario": name, "expected": exp, "observed": r["W_max"],
                     "ci99_half": 0.0, "ok": ok, "note": f"W_min={r['W_min']:.6f}"})

    res, (m, h) = series(replace(base, p_found=0.3), "frac_found1")
    rows.append(_row("V7 p=0.3: частка завершених на ЕОМ-1", 0.3, m, h))

    res, (m, h) = series(base, "throughput")
    rows.append(_row("V8 пропускна здатність = 1/10, 1/с", 0.1, m, h))

    res, (m, h) = series(base, "util_cpu1")
    rows.append(_row("V9 завантаження ЕОМ-1 = 2 с / 10 с", 0.2, m, h))
    res, (m, h) = series(base, "util_cpu2")
    rows.append(_row("V10 завантаження ЕОМ-2 = (1-p)*2 с / 10 с", 0.1, m, h))

    res = replicate(base, n_rep, t_end, warmup, workers=workers)
    ratios = [r["N_mean"] / (r["throughput"] * r["W_mean"]) for r in res]
    m, h = ci99(ratios)
    rows.append(_row("V11 закон Літтла: L / (lambda*W) = 1", 1.0, m, h, tol_abs=0.01))

    sat = replace(base, interarrival_mean=1.5, interarrival_half=0.0, p_found=1.0, search_half=0.0)
    r = simulate(sat, 20000.0, 2000.0, seed=1)
    rows.append(_row("V12 насичення ЕОМ-1 (інтервал 1.5 с, обробка 2 с): пропускна здатність 1/2 с",
                     0.5, r["throughput"], 0.0, tol_abs=0.005))

    hold = replace(base, search_occupies_cpu=True, p_found=1.0, search_half=0.0,
                   interarrival_mean=10.0, interarrival_half=0.0)
    r = simulate(hold, 20000.0, 2000.0, seed=1)
    inv_ok &= all(a == b for _, a, b in r["invariants"])
    rows.append(_row("V13 процесор зайнятий пошуком (20 с): пропускна здатність 1/20 с",
                     0.05, r["throughput"], 0.0, tol_abs=0.002))

    chan = replace(base, interarrival_mean=2.5, interarrival_half=0.0, p_found=0.0, search_half=0.0)
    r = simulate(chan, 20000.0, 2000.0, seed=1)
    inv_ok &= all(a == b for _, a, b in r["invariants"])
    rows.append(_row("V14 насичення каналу (p=0, інтервал 2.5 с, передача 3 с): пропускна здатність 1/3 с",
                     1.0 / 3.0, r["throughput"], 0.0, tol_abs=0.005))

    rows.append({"scenario": "V15 інваріанти (збереження запитів, P-інваріанти, інваріанти каналів) на всіх прогонах",
                 "expected": 1, "observed": int(inv_ok), "ci99_half": 0.0, "ok": inv_ok, "note": ""})

    path = os.path.join(out_dir, "verification_table.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows, path

def print_table(rows):
    print(f"{'Сценарій':<86}{'Очікув.':>10}{'Отрим.':>11}{'+-CI99':>9}  Результат")
    for r in rows:
        print(f"{r['scenario']:<86}{r['expected']:>10.4f}{r['observed']:>11.4f}"
              f"{r['ci99_half']:>9.4f}  {'ВІРНО' if r['ok'] else 'ПОМИЛКА'}")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    event_log("results")
    rows, _ = verification_table("results")
    print_table(rows)