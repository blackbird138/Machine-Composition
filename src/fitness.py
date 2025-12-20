from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import Counter
import math

from .constants import C_MAJOR_SCALE, DURATIONS

def pc(midi_pitch: int) -> int:
    return midi_pitch % 12

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

    def evaluate(self, melody):
        notes = melody.notes
        if not notes:
            melody.fitness_score = 0.0
            melody.chord_progression = []
            return 0.0

        progression, harmony_score, harmony_debug = self._harmony_fitness(notes)

        contour_score = self._melodic_contour(notes)
        rhythmic_score = self._rhythmic_variety(notes)
        cadence_score = self._cadence_melody_side(notes)

        total = harmony_score + contour_score + rhythmic_score + cadence_score

        melody.fitness_score = float(max(0.0, min(100.0, total)))
        melody.chord_progression = progression
        melody.harmony_debug = harmony_debug
        return melody.fitness_score

    def _melodic_contour(self, notes) -> float:
        score = 15.0
        for i in range(len(notes) - 1):
            interval = abs(notes[i + 1].pitch - notes[i].pitch)
            if interval > 12:
                score -= 3.0
            elif interval > 7:
                score -= 1.5
        return max(0.0, score)

    def _rhythmic_variety(self, notes) -> float:
        durations = [n.duration for n in notes]
        unique = len(set(durations))
        denom = max(1, len(DURATIONS))
        return (unique / denom) * 10.0

    def _cadence_melody_side(self, notes) -> float:
        end_pc = pc(notes[-1].pitch)
        if end_pc == self.tonic_pc:
            return 5.0
        if end_pc in {self.tonic_pc, (self.tonic_pc + 4) % 12, (self.tonic_pc + 7) % 12}:
            return 2.0
        return 0.0

    def _harmony_fitness(self, notes) -> Tuple[List[str], float, dict]:
        events, total_beats = self._build_events(notes)

        segments = self._build_segments(total_beats)

        min_dur = min((n.duration for n in notes), default=self.segment_beats)
        short_thresh = 1.5 * float(min_dur)

        emission = []
        seg_note_stats = []
        for (a, b) in segments:
            seg_notes = self._collect_segment_notes(events, a, b)
            seg_note_stats.append(seg_notes)
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
            "per_beat": best_raw / max(1e-9, total_beats),
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
        step = max(1e-9, self.segment_beats)
        while cur < total_beats - 1e-9:
            segs.append((cur, min(total_beats, cur + step)))
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
        per_beat = raw / max(1e-9, total_beats)
        lo, hi = -1.2, 2.2
        x = max(lo, min(hi, per_beat))
        ratio = (x - lo) / (hi - lo)  # 0..1
        return 70.0 * ratio

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
