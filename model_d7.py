from __future__ import annotations
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from petri_net import PetriNet, Simulator, Unif, Const

QUEUES = ("Q1", "C1", "Q2", "R")

@dataclass(frozen=True)
class D7Params:
    interarrival_mean: float = 10.0
    interarrival_half: float = 3.0
    prim_time: float = 2.0
    search_mean: float = 18.0
    search_half: float = 2.0
    channel_time: float = 3.0
    p_found: float = 0.5
    cpu1: int = 1
    cpu2: int = 1
    return_result: bool = True
    prim_uses_cpu: bool = False

def build_net(p: D7Params) -> PetriNet:
    n = PetriNet()
    Pgen = n.place("Pgen", 1)
    P0, B1 = n.place("P0"), n.place("B1")
    Q1, C1, Q2, R = n.place("Q1"), n.place("C1"), n.place("Q2"), n.place("R")
    CPU1, CPU2 = n.place("CPU1", p.cpu1), n.place("CPU2", p.cpu2)
    CH12, CH21 = n.place("CH12", 1), n.place("CH21", 1)
    ARR, Done = n.place("ARR"), n.place("Done")

    n.transition("arrive", Unif(p.interarrival_mean, p.interarrival_half),
                 [(Pgen, 1)], [(Pgen, 1), (P0, 1), (ARR, 1)])
    if p.prim_uses_cpu:
        n.transition("prim", Const(p.prim_time), [(P0, 1), (CPU1, 1)], [(B1, 1), (CPU1, 1)])
    else:
        n.transition("prim", Const(p.prim_time), [(P0, 1)], [(B1, 1)])
    n.transition("loc", Const(0.0), [(B1, 1)], [(Q1, 1)], prob=p.p_found)
    n.transition("rem", Const(0.0), [(B1, 1)], [(C1, 1)], prob=1.0 - p.p_found)
    search = Unif(p.search_mean, p.search_half)
    n.transition("search1", search, [(Q1, 1), (CPU1, 1)], [(CPU1, 1), (Done, 1)])
    n.transition("ch12", Const(p.channel_time), [(C1, 1), (CH12, 1)], [(CH12, 1), (Q2, 1)])
    if p.return_result:
        n.transition("search2", search, [(Q2, 1), (CPU2, 1)], [(CPU2, 1), (R, 1)])
        n.transition("ch21", Const(p.channel_time), [(R, 1), (CH21, 1)], [(CH21, 1), (Done, 1)])
    else:
        n.transition("search2", search, [(Q2, 1), (CPU2, 1)], [(CPU2, 1), (Done, 1)])
    return n

def check_invariants(sim: Simulator, p: D7Params) -> bool:
    pl = sim.net.places
    tr = {t.name: t for t in sim.net.transitions}
    ok = pl["Pgen"].marking + len(tr["arrive"].exits) == 1
    ok &= pl["CH12"].marking + len(tr["ch12"].exits) == 1
    if p.return_result:
        ok &= pl["CH21"].marking + len(tr["ch21"].exits) == 1
    ok &= pl["CPU2"].marking + len(tr["search2"].exits) == p.cpu2
    held1 = len(tr["search1"].exits) + (len(tr["prim"].exits) if p.prim_uses_cpu else 0)
    ok &= pl["CPU1"].marking + held1 == p.cpu1
    in_flight = sum(len(tr[k].exits) for k in ("prim", "loc", "rem", "search1", "ch12", "search2")
                    ) + (len(tr["ch21"].exits) if p.return_result else 0)
    inside = sum(pl[k].marking for k in ("P0", "B1", "Q1", "C1", "Q2", "R")) + in_flight
    ok &= pl["ARR"].marking - pl["Done"].marking == inside
    return bool(ok)

def simulate(p: D7Params, t_end: float, warmup: float = 0.0, seed: int = 1,
             checkpoint_step=None, log_limit: int = 0) -> dict:
    sim = Simulator(build_net(p), seed=seed, stat_start=warmup, log_limit=log_limit, watch=("ARR", "Done"))
    sim.run(t_end, checkpoint_step)
    pl = sim.net.places
    T = sim.stat_time
    n_done = pl["Done"].marking - sim.marking0["Done"]
    area_N = pl["ARR"].area - pl["Done"].area
    queues = QUEUES + (("P0",) if p.prim_uses_cpu else ())
    area_q = sum(pl[k].area for k in queues)
    tr = {t.name: t for t in sim.net.transitions}
    loc, rem = tr["loc"].fired, tr["rem"].fired
    out = {
        "T_stat": T, "n_done": n_done,
        "W": area_N / n_done if n_done else float("nan"),
        "N": area_N / T if T > 0 else float("nan"),
        "X": n_done / T if T > 0 else float("nan"),
        "rho1": 1.0 - pl["CPU1"].area / (p.cpu1 * T),
        "rho2": 1.0 - pl["CPU2"].area / (p.cpu2 * T),
        "Lq": area_q / T,
        "Wq": area_q / n_done if n_done else float("nan"),
        "frac_local": loc / (loc + rem) if loc + rem else float("nan"),
        "queues": {q: (pl[q].area / T,
                       pl[q].area / tr[c].fired if c in tr and tr[c].fired else float("nan"),
                       tr[c].fired / T if c in tr else float("nan"))
                   for q, c in (("Q1", "search1"), ("C1", "ch12"), ("Q2", "search2"), ("R", "ch21"))},
        "invariants": check_invariants(sim, p),
        "checkpoints": sim.checkpoints,
        "marking_end": sim.net.marking(),
        "log": sim.log,
    }
    return out

def _worker(args):
    p, t_end, warmup, seed, step = args
    r = simulate(p, t_end, warmup, seed, step)
    r.pop("log")
    return r

def replicate(p: D7Params, n: int, t_end: float, warmup: float = 0.0, seed0: int = 1000,
              workers: int | None = None, checkpoint_step=None) -> list[dict]:
    jobs = [(p, t_end, warmup, seed0 + i, checkpoint_step) for i in range(n)]
    workers = workers if workers is not None else (os.cpu_count() or 1)
    if workers <= 1:
        return [_worker(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_worker, jobs, chunksize=max(1, n // (workers * 4))))

def analytic_response_no_queue(p: D7Params) -> float:
    remote = p.channel_time + p.search_mean + (p.channel_time if p.return_result else 0.0)
    return p.prim_time + p.p_found * p.search_mean + (1.0 - p.p_found) * remote

__all__ = ["D7Params", "build_net", "simulate", "replicate", "analytic_response_no_queue", "replace"]
