"""
Export melodies to ABC notation and MIDI files.
"""

from .constants import TIME_SIGNATURE, KEY_SIGNATURE


class MelodyExporter:
    """Export melodies to various formats."""
    
    @staticmethod
    def to_abc_file(melody, filename="output.abc", title="Generated Melody"):
        """
        Export melody to ABC notation file.
        
        Args:
            melody (Melody): Melody to export
            filename (str): Output filename
            title (str): Title for the ABC notation
        """
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
        """
        Export melody to MIDI file (requires midiutil).
        
        Args:
            melody (Melody): Melody to export
            filename (str): Output filename
            tempo (int): Tempo in BPM
            
        Returns:
            bool: True if successful, False if midiutil not available
        """
        try:
            from midiutil import MIDIFile
            
            midi = MIDIFile(1)  # One track
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
