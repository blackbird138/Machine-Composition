"""
Musical constants and parameters for the genetic algorithm.
"""

# MIDI pitch mapping
PITCHES = {
    'F3': 53, 'F#3': 54, 'G3': 55, 'G#3': 56, 'A3': 57, 'A#3': 58, 'B3': 59,
    'C4': 60, 'C#4': 61, 'D4': 62, 'D#4': 63, 'E4': 64, 'F4': 65, 'F#4': 66, 
    'G4': 67, 'G#4': 68, 'A4': 69, 'A#4': 70, 'B4': 71,
    'C5': 72, 'C#5': 73, 'D5': 74, 'D#5': 75, 'E5': 76, 'F5': 77, 'F#5': 78, 'G5': 79
}

PITCH_NAMES = list(PITCHES.keys())
PITCH_VALUES = list(PITCHES.values())

# Reverse mapping for ABC notation
MIDI_TO_PITCH = {v: k for k, v in PITCHES.items()}

# Durations in quarter notes
# 0.5 = eighth note, 1.0 = quarter note, 2.0 = half note
DURATIONS = [0.5, 1.0, 2.0]

# Musical parameters
TARGET_BEATS = 16  # 4 measures * 4 beats
TIME_SIGNATURE = "4/4"
KEY_SIGNATURE = "Cmaj"

# C major scale (MIDI values)
C_MAJOR_SCALE = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79]
