from dataclasses import dataclass
from typing import List, Tuple
from collections import Counter
import math

from .constants import C_MAJOR_SCALE, DURATIONS


def pc(p: int) -> int:
    return p % 12


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True)
class Chord:
    name: str
    root_pc: int
    tones: Tuple[int, ...]
    func: str


def build_chords(include_v7: bool = True) -> List[Chord]:
    I   = Chord("C",    0,  (0, 4, 7),     "T")
    ii  = Chord("Dm",   2,  (2, 5, 9),     "PD")
    iii = Chord("Em",   4,  (4, 7, 11),    "T")
    IV  = Chord("F",    5,  (5, 9, 0),     "PD")
    V   = Chord("G",    7,  (7, 11, 2),    "D")
    vi  = Chord("Am",   9,  (9, 0, 4),     "T")
    vii = Chord("Bdim", 11, (11, 2, 5),    "D")
    chords = [I, ii, iii, IV, V, vi, vii]
    if include_v7:
        chords.append(Chord("G7", 7, (7, 11, 2, 5), "D"))
    return chords


class FitnessEvaluator:
    MAX_NOTES = 32

    def __init__(
        self,
        scale=None,
        beats_per_measure: float = 4.0,
        segment_beats: float = 4.0,
        include_v7: bool = True,
    ):
        self.scale = scale if scale else C_MAJOR_SCALE
        self.spcs = {pc(p) for p in self.scale}
        self.tonic = pc(self.scale[0]) if self.scale else 0

        self.bpm = float(beats_per_measure)
        self.seg_beats = float(segment_beats)

        self.chords = build_chords(include_v7=include_v7)
        self.chord_map = {c.name: c for c in self.chords}

        # sigmoid params (pre-calibrated)
        self.contour_c = 30.0
        self.contour_s = 0.1

        self.rhythm_c = 0.50
        self.rhythm_s = 6.0

        self.cadence_c = 0.50
        self.cadence_s = 8.0

        self.harmony_c = 0.60
        self.harmony_s = 2.5

        self.rep_c = 10.0
        self.rep_s = 0.3


    def evaluate(self, melody):
        ns = melody.notes
        if not ns:
            melody.fitness_score = 0.0
            melody.chord_progression = []
            return 0.0

        prog, h_score, _dbg = self._harmony(ns)

        contour = self._contour(ns)
        rep = self._rep_score(ns)
        rhythm = self._rhythm(ns)
        cadence = self._cadence(ns)

        w_h, w_c, w_rp, w_rt, w_cd = 0.6, 0.15, 0.1, 0.1, 0.05

        total = (
            w_h * h_score +
            w_c * contour +
            w_rp * rep +
            w_rt * rhythm +
            w_cd * cadence
        )

        melody.fitness_score = float(total)
        melody.chord_progression = prog
        return melody.fitness_score


    def _contour(self, notes) -> float:
        pen = 0.0
        n = len(notes)
        for i in range(n - 1):
            a = int(notes[i].pitch)
            b = int(notes[i + 1].pitch)
            d = abs(b - a)

            if d > 12:
                pen += 5.0
            elif d > 7:
                pen += 1.0
            elif d > 5:
                pen += 0.5

            ic = d % 12
            if ic == 0:
                pen += 0.0
            elif ic == 7:
                pen += 0.2
            elif ic == 5:
                pen += 0.35
            elif ic == 4:
                pen += 0.45
            elif ic == 3:
                pen += 0.55
            elif ic == 9:
                pen += 0.60
            elif ic == 8:
                pen += 0.70
            elif ic in (2, 10):
                pen += 1.20
            elif ic in (1, 11):
                pen += 1.45
            elif ic == 6:
                pen += 1.80
            else:
                pen += 1.0

        z = self.contour_s * (self.contour_c - pen)
        return sigmoid(z)

    def _rep_score(self, notes) -> float:
        n = len(notes)
        pen = 0.0
        for i in range(n):
            pi = int(notes[i].pitch)
            for j in range(i + 1, n):
                if int(notes[j].pitch) == pi:
                    d = float(j - i)
                    pen += 1.0 / d
        z = self.rep_s * (self.rep_c - pen)
        return sigmoid(z)


    def _rhythm(self, notes) -> float:
        durs = [n.duration for n in notes]
        if not durs:
            return 0.0

        onsets = []
        t = 0.0
        for n in notes:
            onsets.append((t, n.duration))
            t += n.duration

        unique = float(len(set(durs)))
        denom = float(len(DURATIONS)) if len(DURATIONS) > 0 else 1.0
        var = 1.0 - math.exp(-unique / denom)

        bal = self._dur_entropy(durs)

        pen = 0.0
        run = 1
        for i in range(1, len(durs)):
            if durs[i] == durs[i - 1]:
                run += 1
            else:
                if run >= 4:
                    pen += (run - 3) * 1.5
                run = 1
        if run >= 4:
            pen += (run - 3) * 1.5

        mono = math.exp(-0.25 * pen)
        down = self._downbeat(onsets)
        sync = self._sync(onsets)
        sim = self._bar_sim(onsets)

        w_var, w_bal, w_mono, w_down, w_sync, w_sim = 0.18, 0.17, 0.18, 0.18, 0.14, 0.15
        raw = (
            w_var  * var +
            w_bal  * bal +
            w_mono * mono +
            w_down * down +
            w_sync * sync +
            w_sim  * sim
        )

        z = self.rhythm_s * (raw - self.rhythm_c)
        return sigmoid(z)

    def _dur_entropy(self, durs) -> float:
        cnt = Counter(durs)
        tot = float(sum(cnt.values()))
        if tot <= 0.0:
            return 0.0
        probs = [c / tot for c in cnt.values()]
        H = -sum(p * math.log(p + 1e-12) for p in probs)
        K = float(len(DURATIONS)) if len(DURATIONS) > 0 else 1.0
        max_H = math.log(K + 1e-12)
        if max_H <= 1e-12:
            return 0.0
        return H / max_H

    def _downbeat(self, onsets) -> float:
        bonus = 0.0
        for start, dur in onsets:
            beat = start % self.bpm
            if beat < 1e-6 and dur >= 1.0:
                bonus += 1.6
        scale = 8.0
        return 1.0 - math.exp(-bonus / (scale + 1e-12))

    def _sync(self, onsets) -> float:
        bonus = 0.0
        for start, dur in onsets:
            pos = start % self.bpm
            within = pos % 1.0
            if 0.45 < within < 0.55:
                bonus += 0.8
            if within + dur > 1.0 and dur >= 0.5:
                bonus += 1.0
        scale = 6.0
        return 1.0 - math.exp(-bonus / (scale + 1e-12))

    def _bar_sim(self, onsets) -> float:
        if not onsets:
            return 0.0
        bar_len = self.bpm
        slots = int(bar_len * 2)
        bars = [[] for _ in range(4)]
        for start, dur in onsets:
            bi = int(start // bar_len)
            if bi >= 4:
                break
            rel = start - bi * bar_len
            rem = dur
            pos = int(round(rel * 2))
            while rem > 1e-6 and pos < slots:
                bars[bi].append('1' if rem == dur else '0')
                rem -= 0.5
                pos += 1

        pats = []
        for b in bars:
            if len(b) < slots:
                b = b + ['0'] * (slots - len(b))
            else:
                b = b[:slots]
            pats.append(''.join(b))

        sims = []
        for i in range(len(pats)):
            for j in range(i + 1, len(pats)):
                a, b = pats[i], pats[j]
                if not a or not b:
                    continue
                matches = sum(1 for x, y in zip(a, b) if x == y)
                sims.append(matches / len(a))

        if not sims:
            return 0.0
        return sum(sims) / float(len(sims))

    def _cadence(self, notes) -> float:
        end = pc(notes[-1].pitch)
        tonic = self.tonic
        if end == tonic:
            raw = 1.0
        elif end in {tonic, (tonic + 4) % 12, (tonic + 7) % 12}:
            raw = 0.4
        else:
            raw = 0.0
        z = self.cadence_s * (raw - self.cadence_c)
        return sigmoid(z)


    def _harmony(self, notes) -> Tuple[List[str], float, dict]:
        events, total = self._events(notes)
        segs = self._segs(total)

        min_dur = self.seg_beats
        for n in notes:
            d = float(n.duration)
            if d < min_dur:
                min_dur = d
        short_thresh = 1.5 * float(min_dur)

        emission = []
        for a, b in segs:
            s_notes = self._collect(events, a, b)
            emission.append([self._emit(ch, s_notes, short_thresh) for ch in self.chords])

        T = len(segs)
        C = len(self.chords)
        NEG = -1e18
        dp = [[NEG] * C for _ in range(T)]
        prev = [[None] * C for _ in range(T)]

        start_b = [0.0] * C
        for j, ch in enumerate(self.chords):
            if ch.name == "C":
                start_b[j] += 1.0
            if ch.func == "D":
                start_b[j] -= 0.3

        for j in range(C):
            dp[0][j] = emission[0][j] + start_b[j]

        change_cost = 0.35

        for t in range(1, T):
            for j, ch2 in enumerate(self.chords):
                best = NEG
                bi = None
                for i, ch1 in enumerate(self.chords):
                    trans = self._trans(ch1, ch2)
                    cost = change_cost if ch1.name != ch2.name else 0.0
                    val = dp[t - 1][i] + trans - cost + emission[t][j]
                    if val > best:
                        best = val
                        bi = i
                dp[t][j] = best
                prev[t][j] = bi

        end_b = [0.0] * C
        for j, ch in enumerate(self.chords):
            if ch.name == "C":
                end_b[j] += 2.5
            if ch.func == "D":
                end_b[j] -= 0.5

        last = max(range(C), key=lambda j: dp[T - 1][j] + end_b[j])
        best_raw = dp[T - 1][last] + end_b[last]

        idxs = [last]
        for t in range(T - 1, 0, -1):
            idxs.append(prev[t][idxs[-1]])
        idxs.reverse()
        prog = [self.chords[i].name for i in idxs]

        h_score = self._norm_harmony(best_raw, total)
        fit_ratio = self._tone_ratio(events, segs, idxs)
        debug = {
            "raw": best_raw,
            "total_beats": total,
            "per_beat": best_raw / (total + 1e-12),
            "fit_ratio": fit_ratio,
            "segments": [(a, b) for (a, b) in segs],
        }
        return prog, h_score, debug

    def _events(self, notes) -> Tuple[List[Tuple[float, float, int]], float]:
        t = 0.0
        ev = []
        for n in notes:
            s = t
            e = t + float(n.duration)
            ev.append((s, e, int(n.pitch)))
            t = e
        return ev, t

    def _segs(self, total: float) -> List[Tuple[float, float]]:
        segs = []
        cur = 0.0
        step = self.seg_beats if self.seg_beats > 1e-9 else 1e-9
        while cur < total - 1e-9:
            nxt = cur + step
            segs.append((cur, nxt if nxt < total else total))
            cur += step
        if not segs:
            segs = [(0.0, total)]
        return segs

    def _collect(self, events, a, b) -> List[Tuple[int, float]]:
        out = []
        for s, e, p in events:
            overlap = max(0.0, min(e, b) - max(s, a))
            if overlap > 1e-9:
                out.append((p, overlap))
        return out

    def _emit(self, chord: Chord, seg_notes, short_thresh: float) -> float:
        sc = 0.0
        tones = set(chord.tones)
        for pitch, w in seg_notes:
            cls = pc(pitch)
            if cls in tones:
                sc += 2.2 * w
            elif cls in self.spcs:
                if w <= short_thresh:
                    sc += 0.2 * w
                else:
                    sc -= 0.8 * w
            else:
                sc -= 3.2 * w
        return sc

    def _trans(self, c1: Chord, c2: Chord) -> float:
        fb = {
            ("T", "PD"): 1.0,
            ("PD", "D"): 2.0,
            ("D", "T"): 3.0,
            ("T", "D"): 0.8,
            ("T", "T"): 0.3,
            ("PD", "T"): 0.2,
            ("PD", "PD"): 0.0,
            ("D", "D"): 0.0,
            ("D", "PD"): -1.0,
        }
        s = fb.get((c1.func, c2.func), -0.2)
        diff = (c2.root_pc - c1.root_pc) % 12
        if diff in (5, 7):
            s += 0.8
        if c1.func == "D" and c2.name == "C":
            s += 1.2
        return s

    def _norm_harmony(self, raw: float, total_beats: float) -> float:
        per = raw / (total_beats + 1e-12)
        z = self.harmony_s * (per - self.harmony_c)
        return sigmoid(z)

    def _tone_ratio(self, events, segs, idxs) -> float:
        tot = 0.0
        hit = 0.0
        for seg, ci in zip(segs, idxs):
            a, b = seg
            s_notes = self._collect(events, a, b)
            tones = set(self.chords[ci].tones)
            for p, w in s_notes:
                tot += w
                if pc(p) in tones:
                    hit += w
        return 0.0 if tot < 1e-9 else hit / tot
