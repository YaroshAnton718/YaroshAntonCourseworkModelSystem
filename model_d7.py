from __future__ import annotations
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from petri_net import Const, PetriNet, Simulator, Unif

@dataclass(frozen=True)
class D7Params:
    interarrival_mean: float = 10.0
    interarrival_half: float = 3.0
    prim_time: float = 2.0
    search_mean: float = 18.0
    search_half: float = 2.0
    channel_time: float = 3.0
    p_found: float = 0.5
    return_result: bool = True
    channel_single: bool = True
    search_occupies_cpu: bool = False
    cpu_channels: int = 1

    @property
    def lam(self) -> float:
        return 1.0 / self.interarrival_mean

def build_net(p: D7Params) -> PetriNet:
    net = PetriNet()
    gen = net.place("P_gen", 1)
    q1, a1, b1, c1 = (net.place(n) for n in ("Q1", "A1", "B1", "C1"))
    q2, a2, r = (net.place(n) for n in ("Q2", "A2", "R"))
    cpu1 = net.place("CPU1", p.cpu_channels)
    cpu2 = net.place("CPU2", p.cpu_channels)
    done = net.place("Done", sink=True)

    prim, ch = Const(p.prim_time), Const(p.channel_time)
    search = Unif(p.search_mean, p.search_half)
    hold = p.search_occupies_cpu

    net.transition("arrive", Unif(p.interarrival_mean, p.interarrival_half),
                   [(gen, 1)], [(gen, 1, "res"), (q1, 1, "new")])

    net.transition("prim1", prim, [(q1, 1), (cpu1, 1)],
                   [(a1, 1, "carry")] + ([] if hold else [(cpu1, 1, "res")]))
    net.transition("search1", search, [(a1, 1)], [(b1, 1, "carry")])
    free1 = [(cpu1, 1, "res")] if hold else []
    net.transition("found1", Const(0), [(b1, 1)], [(done, 1, "carry")] + free1, prob=p.p_found)
    net.transition("notfound1", Const(0), [(b1, 1)], [(c1, 1, "carry")] + free1,
                   prob=1.0 - p.p_found)

    if p.channel_single:
        chan12 = net.place("CH12", 1)
        net.transition("ch12", ch, [(c1, 1), (chan12, 1)], [(q2, 1, "carry"), (chan12, 1, "res")])
    else:
        net.transition("ch12", ch, [(c1, 1)], [(q2, 1, "carry")])

    net.transition("prim2", prim, [(q2, 1), (cpu2, 1)],
                   [(a2, 1, "carry")] + ([] if hold else [(cpu2, 1, "res")]))
    after2 = r if p.return_result else done
    if hold:
        b2 = net.place("B2")
        net.transition("search2", search, [(a2, 1)], [(b2, 1, "carry")])
        net.transition("finish2", Const(0), [(b2, 1)], [(after2, 1, "carry"), (cpu2, 1, "res")])
    else:
        net.transition("search2", search, [(a2, 1)], [(after2, 1, "carry")])

    if p.return_result:
        if p.channel_single:
            chan21 = net.place("CH21", 1)
            net.transition("ch21", ch, [(r, 1), (chan21, 1)], [(done, 1, "carry"), (chan21, 1, "res")])
        else:
            net.transition("ch21", ch, [(r, 1)], [(done, 1, "carry")])
    return net

def check_invariants(sim: Simulator, p: D7Params) -> list[tuple[str, float, float]]:
    net = sim.net
    P, T = net.places, {t.name: t for t in net.transitions}
    res = [("закон збереження запитів: надійшло - завершено = у системі",
            sim.arrived - sim.completed, net.count_requests())]
    for k, other_places in ((1, ("A1", "B1")), (2, ("A2", "B2"))):
        busy = len(T[f"prim{k}"].exits)
        if p.search_occupies_cpu:
            busy += sum(P[n].marking for n in other_places if n in P)
            busy += len(T[f"search{k}"].exits)
            busy += sum(len(T[n].exits) for n in (f"found{k}", f"notfound{k}", f"finish{k}") if n in T)
        res.append((f"P-інваріант процесора {k}: вільні + зайняті = {p.cpu_channels}",
                    P[f"CPU{k}"].marking + busy, p.cpu_channels))
    if p.channel_single:
        for name, tr in (("CH12", "ch12"), ("CH21", "ch21")):
            if name in P:
                res.append((f"P-інваріант каналу {name}: вільний + зайнятий = 1",
                            P[name].marking + len(T[tr].exits), 1))
    res.append(("генератор: маркер P_gen завжди рівно один", P["P_gen"].marking + len(T["arrive"].exits), 1))
    return res

def simulate(p: D7Params, t_end: float, warmup: float = 0.0, seed=None,
             checkpoint_step=None, log_limit=0, keep_records=False) -> dict:
    sim = Simulator(build_net(p), seed=seed, stat_start=warmup, log_limit=log_limit)
    out = sim.run(t_end, checkpoint_step)
    out["util_cpu1"] = 1.0 - out["place_mean"]["CPU1"] / p.cpu_channels
    out["util_cpu2"] = 1.0 - out["place_mean"]["CPU2"] / p.cpu_channels
    f1, nf1 = out["fired"]["found1"], out["fired"]["notfound1"]
    out["frac_found1"] = f1 / (f1 + nf1) if f1 + nf1 else float("nan")
    pm = out["place_mean"]
    lq = pm["Q1"] + pm["Q2"] + pm["C1"] + pm["R"]
    out["Lq"] = lq
    out["Wq"] = lq / out["throughput"] if out["throughput"] > 0 else float("nan")
    out["invariants"] = check_invariants(sim, p)
    out["checkpoints"] = sim.checkpoints
    out["log"] = sim.log
    if keep_records:
        out["records"] = sim.records
    return out

def _worker(args):
    p, t_end, warmup, seed, step, keep = args
    r = simulate(p, t_end, warmup, seed, step, keep_records=keep)
    r.pop("log")
    return r

def replicate(p: D7Params, n: int, t_end: float, warmup: float = 0.0, seed0: int = 1000,
              workers: int | None = None, checkpoint_step=None, keep_records=False) -> list[dict]:
    jobs = [(p, t_end, warmup, seed0 + i, checkpoint_step, keep_records) for i in range(n)]
    workers = workers if workers is not None else (os.cpu_count() or 1)
    if workers <= 1:
        return [_worker(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_worker, jobs, chunksize=max(1, n // (workers * 4))))

def analytic_response_no_queue(p: D7Params) -> float:
    stage = p.prim_time + p.search_mean
    second = p.channel_time + stage + (p.channel_time if p.return_result else 0.0)
    return stage + (1.0 - p.p_found) * second

__all__ = ["D7Params", "build_net", "simulate", "replicate", "analytic_response_no_queue",
           "check_invariants", "replace"]