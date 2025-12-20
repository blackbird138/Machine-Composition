from .constants import TIME_SIGNATURE, KEY_SIGNATURE

class MelodyExporter:
    @staticmethod
    def to_abc_file(melody, filename="output.abc", title="Generated Melody"):
        abc_content = f"""X:1
T:{title}
M:{TIME_SIGNATURE}
L:1/8
K:{KEY_SIGNATURE}
{melody.to_abc()}
"""
        with open(filename, 'w') as f:
            f.write(abc_content)
        print(f"ABC notation saved to {filename}")

    @staticmethod
    def to_midi_file(melody, filename="output.mid", tempo=120):
        try:
            from midiutil import MIDIFile
            
            midi = MIDIFile(1)
            track = 0
            channel = 0
            volume = 100
            
            midi.addTempo(track, 0, tempo)
            
            time = 0
            for note in melody.notes:
                midi.addNote(track, channel, note.pitch, time, note.duration, volume)
                time += note.duration
            
            with open(filename, 'wb') as f:
                midi.writeFile(f)
            print(f"MIDI file saved to {filename}")
            return True
            
        except ImportError:
            print("midiutil not installed. Skipping MIDI export.")
            print("To install: pip install midiutil")
            return False
