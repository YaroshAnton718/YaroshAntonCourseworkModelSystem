from __future__ import annotations
import heapq
import math
import random
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
    def __init__(self, name: str, initial: int = 0):
        self.name = name
        self.marking = initial
        self.area = 0.0
        self.area_all = 0.0

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
        self.exits: list[float] = []
        self.busy_area = 0.0
        self.fired = 0

    def enabled(self) -> bool:
        return all(p.marking >= k for p, k in self.inputs)

    def next_exit(self) -> float:
        return self.exits[0] if self.exits else INF

class PetriNet:
    def __init__(self):
        self.places: dict[str, Place] = {}
        self.transitions: list[Transition] = []

    def place(self, name, initial=0) -> Place:
        p = Place(name, initial)
        self.places[name] = p
        return p

    def transition(self, name, delay, inputs, outputs, priority=0, prob=1.0) -> Transition:
        t = Transition(name, delay, inputs, outputs, priority, prob)
        self.transitions.append(t)
        return t

    def marking(self) -> dict:
        return {n: p.marking for n, p in self.places.items()}

class Simulator:
    def __init__(self, net: PetriNet, seed=None, stat_start=0.0, log_limit=0, watch=()):
        self.net = net
        self.rng = random.Random(seed)
        self.t = 0.0
        self.stat_start = stat_start
        self.stat_time = 0.0
        self.marking0: dict | None = None
        self.watch = list(watch)
        self.checkpoints: list = []
        self.log: list[str] = []
        self.log_limit = log_limit

    def _log(self, kind, tr, extra=""):
        if len(self.log) < self.log_limit:
            m = ", ".join(f"{n}={p.marking}" for n, p in self.net.places.items() if p.marking)
            self.log.append(f"t={self.t:10.4f}  {kind:<6} {tr.name:<9} M: {m}{extra}")

    def _accumulate(self, t0, t1):
        dt = t1 - t0
        if dt <= 0:
            return
        for p in self.net.places.values():
            p.area_all += p.marking * dt
        if t1 > self.stat_start:
            eff = t1 - max(t0, self.stat_start)
            self.stat_time += eff
            for p in self.net.places.values():
                p.area += p.marking * eff
            for tr in self.net.transitions:
                tr.busy_area += len(tr.exits) * eff

    def _exit_all(self):
        for tr in self.net.transitions:
            while tr.exits and tr.exits[0] <= self.t:
                heapq.heappop(tr.exits)
                for place, k in tr.outputs:
                    place.marking += k
                self._log("вихід", tr)

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
        for place, k in tr.inputs:
            place.marking -= k
        t_out = self.t + tr.delay(self.rng)
        heapq.heappush(tr.exits, t_out)
        if self.t >= self.stat_start:
            tr.fired += 1
        self._log("вхід", tr, f"   -> вихід у t={t_out:.4f}")

    def run(self, t_end: float, checkpoint_step: float | None = None):
        trs = self.net.transitions
        next_cp = checkpoint_step if checkpoint_step else INF
        pending_start = self.stat_start > 0.0
        if not pending_start:
            self.marking0 = self.net.marking()
        self._enter_all()
        while True:
            t_ev = min(tr.next_exit() for tr in trs)
            t_next = min(t_ev, next_cp, t_end)
            if pending_start:
                t_next = min(t_next, self.stat_start)
            self._accumulate(self.t, t_next)
            self.t = t_next
            if pending_start and self.t >= self.stat_start:
                self.marking0 = self.net.marking()
                pending_start = False
            if self.t == next_cp:
                self.checkpoints.append((self.t, {n: self.net.places[n].area_all for n in self.watch}))
                next_cp += checkpoint_step
            if self.t >= t_end:
                break
            if t_ev <= self.t:
                self._exit_all()
                self._enter_all()
        return self.t