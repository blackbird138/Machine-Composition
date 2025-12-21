from dataclasses import dataclass
from typing import List, Tuple
from collections import Counter
import math

from .constants import C_MAJOR_SCALE, DURATIONS


def pc(midi_pitch: int) -> int:
    return midi_pitch % 12


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True)
class Chord:
    name: str
    root_pc: int
    tones: Tuple[int, ...]
    function: str


def build_c_major_chords(include_v7: bool = True) -> List[Chord]:
    I   = Chord("C",    0,  (0, 4, 7),     "T")
    ii  = Chord("Dm",   2,  (2, 5, 9),     "PD")
    iii = Chord("Em",   4,  (4, 7, 11),    "T")
    IV  = Chord("F",    5,  (5, 9, 0),     "PD")
    V   = Chord("G",    7,  (7, 11, 2),    "D")
    vi  = Chord("Am",   9,  (9, 0, 4),     "T")
    vii = Chord("Bdim", 11, (11, 2, 5),    "D")

    chords = [I, ii, iii, IV, V, vi, vii]
    if include_v7:
        V7 = Chord("G7", 7, (7, 11, 2, 5), "D")
        chords.append(V7)
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
        self.scale_pcs = {pc(p) for p in self.scale}
        self.tonic_pc = pc(self.scale[0]) if self.scale else 0

        self.beats_per_measure = float(beats_per_measure)
        self.segment_beats = float(segment_beats)

        self.chords = build_c_major_chords(include_v7=include_v7)
        self.chord_by_name = {c.name: c for c in self.chords}
        self.CONTOUR_CENTER = 30.0
        self.CONTOUR_SCALE = 0.1

        self.RHYTHM_CENTER = 0.50
        self.RHYTHM_SCALE = 6.0

        self.CADENCE_CENTER = 0.50
        self.CADENCE_SCALE = 8.0

        self.HARMONY_CENTER = 0.60
        self.HARMONY_SCALE = 2.5

        self.REP_CENTER = 10.0
        self.REP_SCALE = 0.3


    def evaluate(self, melody):
        notes = melody.notes
        if not notes:
            melody.fitness_score = 0.0
            melody.chord_progression = []
            return 0.0

        progression, harmony_score, harmony_debug = self._harmony_fitness(notes)

        contour_score = self._melodic_contour(notes)
        repetition_score = self._repetition_penalty_score(notes)
        rhythmic_score = self._rhythmic_interest(notes)
        cadence_score = self._cadence_melody_side(notes)

        w_harmony = 0.6
        w_contour = 0.15
        w_repeat  = 0.1
        w_rhythm  = 0.1
        w_cadence = 0.05

        total = (
            w_harmony * harmony_score +
            w_contour * contour_score +
            w_repeat  * repetition_score +
            w_rhythm  * rhythmic_score +
            w_cadence * cadence_score
        )


        melody.fitness_score = float(total)
        melody.chord_progression = progression
        melody.harmony_debug = harmony_debug
        return melody.fitness_score

    def _melodic_contour(self, notes) -> float:
        penalty = 0.0
        n = len(notes)

        for i in range(n - 1):
            a = int(notes[i].pitch)
            b = int(notes[i + 1].pitch)

            abs_int = abs(b - a)

            if abs_int > 12:
                penalty += 5.0
            elif abs_int > 7:
                penalty += 1.0
            elif abs_int > 5:
                penalty += 0.5

            ic = abs_int % 12
            if ic == 0:
                penalty += 0.0
            elif ic == 7:
                penalty += 0.2
            elif ic == 5:
                penalty += 0.35
            elif ic == 4:
                penalty += 0.45
            elif ic == 3:
                penalty += 0.55
            elif ic == 9:
                penalty += 0.60
            elif ic == 8:
                penalty += 0.70
            elif ic == 2:
                penalty += 1.20
            elif ic == 10:
                penalty += 1.20
            elif ic == 1:
                penalty += 1.45
            elif ic == 11:
                penalty += 1.45
            elif ic == 6:
                penalty += 1.80
            else:
                penalty += 1.0

        z = self.CONTOUR_SCALE * (self.CONTOUR_CENTER - penalty)
        return sigmoid(z)

    def _repetition_penalty_score(self, notes) -> float:
        n = len(notes)
        penalty = 0.0

        for i in range(n):
            pi = int(notes[i].pitch)
            for j in range(i + 1, n):
                if int(notes[j].pitch) == pi:
                    d = float(j - i)
                    penalty += 1.0 / d

        z = self.REP_SCALE * (self.REP_CENTER - penalty)
        return sigmoid(z)


    def _rhythmic_interest(self, notes) -> float:
        durations = [n.duration for n in notes]
        if not durations:
            return 0.0

        onsets = []
        t = 0.0
        for n in notes:
            onsets.append((t, n.duration))
            t += n.duration

        unique = float(len(set(durations)))
        denom = float(len(DURATIONS)) if len(DURATIONS) > 0 else 1.0
        variety01 = 1.0 - math.exp(-unique / denom)

        balance01 = self._duration_balance_entropy01(durations)

        penalty = 0.0
        run_len = 1
        for i in range(1, len(durations)):
            if durations[i] == durations[i - 1]:
                run_len += 1
            else:
                if run_len >= 4:
                    penalty += (run_len - 3) * 1.5
                run_len = 1
        if run_len >= 4:
            penalty += (run_len - 3) * 1.5

        monotony01 = math.exp(-0.25 * penalty)

        downbeat01 = self._downbeat_bonus01(onsets)

        sync01 = self._syncopation_bonus01(onsets)

        sim01 = self._bar_rhythm_similarity01(onsets)

        w_var, w_bal, w_mono, w_down, w_sync, w_sim = 0.18, 0.17, 0.18, 0.18, 0.14, 0.15
        raw01 = (
            w_var  * variety01 +
            w_bal  * balance01 +
            w_mono * monotony01 +
            w_down * downbeat01 +
            w_sync * sync01 +
            w_sim  * sim01
        )

        z = self.RHYTHM_SCALE * (raw01 - self.RHYTHM_CENTER)
        return sigmoid(z)

    def _duration_balance_entropy01(self, durations) -> float:
        counts = Counter(durations)
        total = float(sum(counts.values()))
        if total <= 0.0:
            return 0.0

        probs = [c / total for c in counts.values()]
        H = -sum(p * math.log(p + 1e-12) for p in probs)

        K = float(len(DURATIONS)) if len(DURATIONS) > 0 else 1.0
        max_H = math.log(K + 1e-12)

        if max_H <= 1e-12:
            return 0.0

        return H / max_H

    def _downbeat_bonus01(self, onsets) -> float:
        bonus = 0.0
        for start, dur in onsets:
            beat_in_measure = start % self.beats_per_measure
            if beat_in_measure < 1e-6 and dur >= 1.0:
                bonus += 1.6

        scale = 8.0
        return 1.0 - math.exp(-bonus / (scale + 1e-12))

    def _syncopation_bonus01(self, onsets) -> float:
        bonus = 0.0
        for start, dur in onsets:
            beat_pos = start % self.beats_per_measure
            within_beat = beat_pos % 1.0

            if 0.45 < within_beat < 0.55:
                bonus += 0.8

            if within_beat + dur > 1.0 and dur >= 0.5:
                bonus += 1.0

        scale = 6.0
        return 1.0 - math.exp(-bonus / (scale + 1e-12))

    def _bar_rhythm_similarity01(self, onsets) -> float:
        if not onsets:
            return 0.0

        bar_len = self.beats_per_measure
        slots_per_bar = int(bar_len * 2)
        bars = [[] for _ in range(4)]

        for start, dur in onsets:
            bar_idx = int(start // bar_len)
            if bar_idx >= 4:
                break
            rel = start - bar_idx * bar_len
            remaining = dur
            pos = int(round(rel * 2))
            while remaining > 1e-6 and pos < slots_per_bar:
                bars[bar_idx].append('1' if remaining == dur else '0')
                remaining -= 0.5
                pos += 1

        bar_patterns = []
        for b in bars:
            if len(b) < slots_per_bar:
                b = b + ['0'] * (slots_per_bar - len(b))
            else:
                b = b[:slots_per_bar]
            bar_patterns.append(''.join(b))

        sims = []
        for i in range(len(bar_patterns)):
            for j in range(i + 1, len(bar_patterns)):
                a, b = bar_patterns[i], bar_patterns[j]
                if not a or not b:
                    continue
                matches = sum(1 for x, y in zip(a, b) if x == y)
                sims.append(matches / len(a))

        if not sims:
            return 0.0

        return sum(sims) / float(len(sims))

    def _cadence_melody_side(self, notes) -> float:
        end_pc = pc(notes[-1].pitch)

        if end_pc == self.tonic_pc:
            raw = 1.0
        elif end_pc in {self.tonic_pc, (self.tonic_pc + 4) % 12, (self.tonic_pc + 7) % 12}:
            raw = 0.4
        else:
            raw = 0.0

        z = self.CADENCE_SCALE * (raw - self.CADENCE_CENTER)
        return sigmoid(z)

    def _harmony_fitness(self, notes) -> Tuple[List[str], float, dict]:
        events, total_beats = self._build_events(notes)
        segments = self._build_segments(total_beats)

        min_dur = self.segment_beats
        for n in notes:
            d = float(n.duration)
            if d < min_dur:
                min_dur = d
        short_thresh = 1.5 * float(min_dur)

        emission = []
        for (a, b) in segments:
            seg_notes = self._collect_segment_notes(events, a, b)
            emission.append([self._emission_score(ch, seg_notes, short_thresh) for ch in self.chords])

        T = len(segments)
        C = len(self.chords)
        NEG = -1e18
        dp = [[NEG] * C for _ in range(T)]
        prev = [[None] * C for _ in range(T)]

        start_bonus = [0.0] * C
        for j, ch in enumerate(self.chords):
            if ch.name == "C":
                start_bonus[j] += 1.0
            if ch.function == "D":
                start_bonus[j] -= 0.3

        for j in range(C):
            dp[0][j] = emission[0][j] + start_bonus[j]

        change_cost = 0.35

        for t in range(1, T):
            for j, ch2 in enumerate(self.chords):
                best_val = NEG
                best_i = None
                for i, ch1 in enumerate(self.chords):
                    trans = self._transition_score(ch1, ch2)
                    cost = change_cost if ch1.name != ch2.name else 0.0
                    val = dp[t - 1][i] + trans - cost + emission[t][j]
                    if val > best_val:
                        best_val = val
                        best_i = i
                dp[t][j] = best_val
                prev[t][j] = best_i

        end_bonus = [0.0] * C
        for j, ch in enumerate(self.chords):
            if ch.name == "C":
                end_bonus[j] += 2.5
            if ch.function == "D":
                end_bonus[j] -= 0.5

        last = max(range(C), key=lambda j: dp[T - 1][j] + end_bonus[j])
        best_raw = dp[T - 1][last] + end_bonus[last]

        idxs = [last]
        for t in range(T - 1, 0, -1):
            idxs.append(prev[t][idxs[-1]])
        idxs.reverse()
        progression = [self.chords[i].name for i in idxs]

        harmony_score = self._normalize_harmony(best_raw, total_beats)

        fit_ratio = self._chord_tone_ratio(events, segments, idxs)
        debug = {
            "raw": best_raw,
            "total_beats": total_beats,
            "per_beat": best_raw / (total_beats + 1e-12),
            "fit_ratio": fit_ratio,
            "segments": [(a, b) for (a, b) in segments],
        }

        return progression, harmony_score, debug

    def _build_events(self, notes) -> Tuple[List[Tuple[float, float, int]], float]:
        t = 0.0
        events = []
        for n in notes:
            start = t
            end = t + float(n.duration)
            events.append((start, end, int(n.pitch)))
            t = end
        return events, t

    def _build_segments(self, total_beats: float) -> List[Tuple[float, float]]:
        segs = []
        cur = 0.0

        step = self.segment_beats if self.segment_beats > 1e-9 else 1e-9

        while cur < total_beats - 1e-9:
            nxt = cur + step
            segs.append((cur, nxt if nxt < total_beats else total_beats))
            cur += step

        if not segs:
            segs = [(0.0, total_beats)]
        return segs

    def _collect_segment_notes(
        self,
        events: List[Tuple[float, float, int]],
        seg_start: float,
        seg_end: float
    ) -> List[Tuple[int, float]]:
        seg_notes = []
        for s, e, p in events:
            overlap = max(0.0, min(e, seg_end) - max(s, seg_start))
            if overlap > 1e-9:
                seg_notes.append((p, overlap))
        return seg_notes

    def _emission_score(self, chord: Chord, seg_notes: List[Tuple[int, float]], short_thresh: float) -> float:
        score = 0.0
        chord_tones = set(chord.tones)

        for pitch, w in seg_notes:
            pclass = pc(pitch)
            if pclass in chord_tones:
                score += 2.2 * w
            elif pclass in self.scale_pcs:
                if w <= short_thresh:
                    score += 0.2 * w
                else:
                    score -= 0.8 * w
            else:
                score -= 3.2 * w
        return score

    def _transition_score(self, c1: Chord, c2: Chord) -> float:
        func_bonus = {
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

        s = func_bonus.get((c1.function, c2.function), -0.2)

        diff = (c2.root_pc - c1.root_pc) % 12
        if diff in (5, 7):
            s += 0.8

        if c1.function == "D" and c2.name == "C":
            s += 1.2

        return s

    def _normalize_harmony(self, raw: float, total_beats: float) -> float:
        per_beat = raw / (total_beats + 1e-12)
        z = self.HARMONY_SCALE * (per_beat - self.HARMONY_CENTER)
        return sigmoid(z)

    def _chord_tone_ratio(self, events, segments, chord_idxs) -> float:
        total = 0.0
        hit = 0.0
        for (seg, ci) in zip(segments, chord_idxs):
            a, b = seg
            seg_notes = self._collect_segment_notes(events, a, b)
            chord = self.chords[ci]
            tones = set(chord.tones)
            for p, w in seg_notes:
                total += w
                if pc(p) in tones:
                    hit += w
        return 0.0 if total < 1e-9 else hit / total
