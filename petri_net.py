from __future__ import annotations
import heapq
import math
import random
from collections import deque
from dataclasses import dataclass

INF = math.inf

class Delay:
    def __call__(self, rng: random.Random) -> float:
        raise NotImplementedError

@dataclass(frozen=True)
class Const(Delay):
    d: float

    def __call__(self, rng):
        return self.d

    def __str__(self):
        return f"{self.d:g}"

@dataclass(frozen=True)
class Unif(Delay):
    d: float
    s: float

    def __call__(self, rng):
        return rng.uniform(self.d - self.s, self.d + self.s)

    def __str__(self):
        return f"unif({self.d:g}, {self.s:g})"

@dataclass(frozen=True)
class Exp(Delay):
    d: float

    def __call__(self, rng):
        return rng.expovariate(1.0 / self.d)

    def __str__(self):
        return f"exp({self.d:g})"

class Place:
    def __init__(self, name: str, initial: int = 0, sink: bool = False):
        self.name = name
        self.sink = sink
        self.tokens: deque = deque([None] * initial)
        self.area = 0.0

    @property
    def marking(self) -> int:
        return len(self.tokens)

class Transition:
    def __init__(self, name, delay, inputs, outputs, priority=0, prob=1.0):
        if not inputs:
            raise ValueError(f"Перехід {name} без вхідних позицій")
        self.name = name
        self.delay = delay
        self.inputs = inputs
        self.outputs = outputs
        self.priority = priority
        self.prob = prob
        self.exits: list = []
        self.busy_area = 0.0
        self.fired = 0

    def enabled(self) -> bool:
        return all(p.marking >= k for p, k in self.inputs)

    def next_exit(self) -> float:
        return self.exits[0][0] if self.exits else INF

class PetriNet:
    def __init__(self):
        self.places: dict[str, Place] = {}
        self.transitions: list[Transition] = []

    def place(self, name, initial=0, sink=False) -> Place:
        p = Place(name, initial, sink)
        self.places[name] = p
        return p

    def transition(self, name, delay, inputs, outputs, priority=0, prob=1.0):
        t = Transition(name, delay, inputs, outputs, priority, prob)
        self.transitions.append(t)
        return t

    def marking(self) -> dict:
        return {n: p.marking for n, p in self.places.items()}

    def count_requests(self) -> int:
        n = sum(1 for p in self.places.values() for tok in p.tokens if tok is not None)
        n += sum(1 for t in self.transitions for e in t.exits if e[2] is not None)
        return n

class Simulator:
    def __init__(self, net: PetriNet, seed=None, stat_start=0.0, log_limit=0):
        self.net = net
        self.rng = random.Random(seed)
        self.t = 0.0
        self.stat_start = stat_start
        self.seq = 0
        self.arrived = 0
        self.completed = 0
        self.records: list = []
        self.area_N = 0.0
        self.area_all = 0.0
        self.stat_time = 0.0
        self.checkpoints: list = []
        self.log: list[str] = []
        self.log_limit = log_limit

    def _log(self, kind, tr, extra=""):
        if len(self.log) < self.log_limit:
            m = ", ".join(f"{n}={p.marking}" for n, p in self.net.places.items() if p.marking)
            self.log.append(f"t={self.t:10.4f}  {kind:<6} {tr.name:<10} M: {m}{extra}")

    def _accumulate(self, t0, t1):
        dt = t1 - t0
        if dt <= 0:
            return
        n = self.arrived - self.completed
        self.area_all += n * dt
        lo = max(t0, self.stat_start)
        if t1 > lo:
            eff = t1 - lo
            self.area_N += n * eff
            self.stat_time += eff
            for p in self.net.places.values():
                p.area += p.marking * eff
            for tr in self.net.transitions:
                tr.busy_area += len(tr.exits) * eff

    def _exit_all(self):
        trs = self.net.transitions[:]
        self.rng.shuffle(trs)
        for tr in trs:
            batch = []
            while tr.exits and tr.exits[0][0] <= self.t:
                batch.append(heapq.heappop(tr.exits))
            self.rng.shuffle(batch)
            for _, _, carried in batch:
                for place, k, mode in tr.outputs:
                    for _ in range(k):
                        if mode == "new":
                            self.arrived += 1
                            self._deposit(place, self.t)
                        elif mode == "carry":
                            self._deposit(place, carried)
                        else:
                            self._deposit(place, None)
                self._log("вихід", tr)

    def _deposit(self, place: Place, token):
        if place.sink:
            if token is not None:
                self.completed += 1
                if self.t >= self.stat_start:
                    self.records.append((token, self.t, self.t - token))
        else:
            place.tokens.append(token)

    def _enter_all(self):
        trs = self.net.transitions
        while True:
            enabled = [tr for tr in trs if tr.prob > 0 and tr.enabled()]
            if not enabled:
                break
            top = max(tr.priority for tr in enabled)
            cand = [tr for tr in enabled if tr.priority == top]
            tr = self.rng.choices(cand, weights=[c.prob for c in cand])[0]
            self._fire(tr)

    def _fire(self, tr: Transition):
        carried = None
        for place, k in tr.inputs:
            for _ in range(k):
                tok = place.tokens.popleft()
                if tok is not None and carried is None:
                    carried = tok
        self.seq += 1
        t_out = self.t + tr.delay(self.rng)
        heapq.heappush(tr.exits, (t_out, self.seq, carried))
        if self.t >= self.stat_start:
            tr.fired += 1
        self._log("вхід", tr, f"   -> вихід у t={t_out:.4f}")

    def run(self, t_end: float, checkpoint_step: float | None = None):
        trs = self.net.transitions
        next_cp = checkpoint_step if checkpoint_step else INF
        self._enter_all()
        while True:
            t_ev = min(tr.next_exit() for tr in trs)
            t_next = min(t_ev, next_cp, t_end)
            self._accumulate(self.t, t_next)
            self.t = t_next
            if self.t == next_cp:
                self.checkpoints.append((self.t, self.area_all))
                next_cp += checkpoint_step
            if self.t >= t_end:
                break
            if t_ev <= self.t:
                self._exit_all()
                self._enter_all()
        return self.results()

    def results(self) -> dict:
        T = self.stat_time
        w = [r[2] for r in self.records]
        n = len(w)
        mean = sum(w) / n if n else float("nan")
        var = sum((x - mean) ** 2 for x in w) / (n - 1) if n > 1 else float("nan")
        return {
            "T_stat": T,
            "arrived": self.arrived,
            "completed": self.completed,
            "n_obs": n,
            "W_mean": mean,
            "W_std": math.sqrt(var) if n > 1 else float("nan"),
            "W_min": min(w) if n else float("nan"),
            "W_max": max(w) if n else float("nan"),
            "N_mean": self.area_N / T if T > 0 else float("nan"),
            "throughput": n / T if T > 0 else float("nan"),
            "place_mean": {k: p.area / T for k, p in self.net.places.items()},
            "busy_mean": {t.name: t.busy_area / T for t in self.net.transitions},
            "fired": {t.name: t.fired for t in self.net.transitions},
        }