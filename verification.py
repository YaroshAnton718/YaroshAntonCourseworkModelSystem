from __future__ import annotations
import math
import statistics as st
from scipy import stats
from model_d7 import D7Params, replace, replicate, simulate, analytic_response_no_queue

def ci(vals, conf=0.95):
    n = len(vals)
    m = st.fmean(vals)
    if n < 2:
        return m, 0.0
    h = stats.t.ppf(0.5 + conf / 2.0, n - 1) * st.stdev(vals) / math.sqrt(n)
    return m, h

def ci95(vals):
    return ci(vals, 0.95)

def variance_simple(vals):
    n = len(vals)
    m = sum(vals) / n
    return (sum(v * v for v in vals) - n * m * m) / (n - 1)

def scenarios():
    b = D7Params()
    far = dict(interarrival_mean=300.0, interarrival_half=90.0)
    S = []
    add = lambda *a: S.append(a)
    add("V1", "інтервал 300 с, p = 0,5", replace(b, **far), 3_000_000, 2000, 20, "W", 23.0, None, "ci")
    add("V2", "інтервал 300 с, p = 1", replace(b, p_found=1.0, **far), 3_000_000, 2000, 20, "W", 20.0, None, "ci")
    add("V3", "інтервал 300 с, p = 0", replace(b, p_found=0.0, **far), 3_000_000, 2000, 20, "W", 26.0, None, "ci")
    add("V4", "інтервал 300 с, p = 0,5, без зворотної передачі",
        replace(b, return_result=False, **far), 3_000_000, 2000, 20, "W", 21.5, None, "ci")
    add("V5", "інтервал 300 с, p = 1, усі затримки сталі",
        replace(b, p_found=1.0, interarrival_mean=300.0, interarrival_half=0.0, search_half=0.0),
        3100, 0.0, 1, "W", 20.0, 1e-9, "abs")
    add("V6", "інтервал 300 с, p = 0, усі затримки сталі",
        replace(b, p_found=0.0, interarrival_mean=300.0, interarrival_half=0.0, search_half=0.0),
        3100, 0.0, 1, "W", 26.0, 1e-9, "abs")
    add("V7", "базові параметри, p = 0,3", replace(b, p_found=0.3), 40_000, 2000, 20, "frac_local", 0.3, None, "ci")
    add("V8", "базові параметри (lambda = 0,1 1/с)", b, 40_000, 2000, 20, "X", 0.1, None, "ci")
    add("V9", "базові параметри", b, 40_000, 2000, 20, "rho1", 0.9, None, "ci")
    add("V10", "базові параметри", b, 40_000, 2000, 20, "rho2", 0.9, None, "ci")
    add("V11", "базові параметри", b, 40_000, 2000, 20, "N/(lambda*W)", 1.0, None, "ci")
    add("V12", "інтервал 1,5 с (сталий), p = 1, пошук 18 с (сталий)",
        replace(b, p_found=1.0, interarrival_mean=1.5, interarrival_half=0.0, search_half=0.0),
        20_000, 2000, 1, "X", 1 / 18, 0.0005, "abs")
    add("V13", "інтервал 1,5 с (сталий), p = 0, пошук 18 с (сталий)",
        replace(b, p_found=0.0, interarrival_mean=1.5, interarrival_half=0.0, search_half=0.0),
        20_000, 2000, 1, "X", 1 / 18, 0.0005, "abs")
    add("V14", "інтервал 1,5 с (сталий), p = 0, пошук 1 с (сталий)",
        replace(b, p_found=0.0, interarrival_mean=1.5, interarrival_half=0.0, search_mean=1.0, search_half=0.0),
        20_000, 2000, 1, "X", 1 / 3, 0.0005, "abs")
    add("V15", "первинна обробка займає процесор, інтервал 1 с, p = 0",
        replace(b, p_found=0.0, prim_uses_cpu=True, interarrival_mean=1.0, interarrival_half=0.0,
                search_mean=1.0, search_half=0.0, channel_time=1.0),
        20_000, 2000, 1, "X", 0.5, 0.0005, "abs")
    add("V16", "2 процесори на комп'ютері 1, p = 1, інтервал 1,5 с, пошук 18 с",
        replace(b, p_found=1.0, cpu1=2, interarrival_mean=1.5, interarrival_half=0.0, search_half=0.0),
        20_000, 2000, 1, "X", 2 / 18, 0.0005, "abs")
    return S

def run_verification(workers=None, seed0=500):
    rows = []
    all_inv = True
    for i, (sid, desc, p, T, warm, n, var, exp, tol, mode) in enumerate(scenarios()):
        res = replicate(p, n, T, warm, seed0=seed0 + 1000 * i, workers=workers)
        if var == "N/(lambda*W)":
            lam = 1.0 / p.interarrival_mean
            vals = [r["N"] / (lam * r["W"]) for r in res]
        else:
            vals = [r[var] for r in res]
        all_inv &= all(r["invariants"] for r in res)
        m, h = ci(vals, 0.99)
        if mode == "ci":
            ok = abs(m - exp) <= h
            tolcol = h
        else:
            ok = abs(m - exp) <= tol
            tolcol = tol
        rows.append((sid, desc, T, n, var, exp, m, tolcol, "вірно" if ok else "НЕВІРНО"))
    rows.append(("V17", "усі прогони попередніх сценаріїв", None, None,
                 "збереження запитів, P-інваріанти ресурсів", 1, 1 if all_inv else 0, 0, "вірно" if all_inv else "НЕВІРНО"))
    return rows

def event_log(path, seed=7, total=60):
    p1 = D7Params()
    first = simulate(p1, 400.0, 0.0, seed, log_limit=total)["log"][:total]
    p2 = replace(p1, p_found=0.0, interarrival_mean=2.5, interarrival_half=0.0, search_half=0.0)
    second = simulate(p2, 60.0, 0.0, seed, log_limit=48)["log"][:48]
    with open(path, "w", encoding="utf-8") as f:
        f.write("Фрагмент 1. Протокол подій (D7, параметри за замовчуванням, зерно ГВЧ = %d)\n" % seed)
        f.write("вхід  - запуск переходу (маркери вилучено з вхідних позицій)\n")
        f.write("вихід - завершення затримки (маркери додано у вихідні позиції)\n")
        f.write("M - позиції з ненульовим маркуванням (CPU1, CPU2, CH12, CH21 - вільні ресурси)\n\n")
        f.write("\n".join(first) + "\n\n")
        f.write("Фрагмент 2. p = 0, інтервал надходження 2,5 с, усі затримки сталі (видно чергу до каналу 1->2)\n\n")
        f.write("\n".join(second) + "\n")
    return first, second

def print_verification(rows):
    print(f"{'N':<5}{'Опис':<62}{'T':>10}{'n':>4}  {'Змінна':<12}{'Очікув.':>10}{'Отрим.':>10}{'+-':>9}  Висновок")
    for sid, desc, T, n, var, exp, m, h, ok in rows:
        print(f"{sid:<5}{desc[:60]:<62}{(T if T is not None else '-')!s:>10}{(n if n is not None else '-')!s:>4}  "
              f"{var[:12]:<12}{exp:>10.4f}{m:>10.4f}{h:>9.4f}  {ok}")
