# Music theory helper functions
# We define pitch to be the MIDI number, and note to be a string such as
# "d#", "cn", or "eb" with no octave information

BASE_NOTES = [n + "n" for n in list("cdefgab")]
NOTES = ["cn", "c#", "dn", "d#", "en", "fn",
         "f#", "gn", "g#", "an", "a#", "bn"]
CMAJOR = dict((n + "n", n + "n") for n in list("cdefgab"))
def pitch_note(pitch_num):
    return NOTES[pitch_num % 12]
def note_pitch(note):
    """ Returns pitch in an arbitrary octave """
    return 60 + NOTES.index(note)

def clef_note(clef, staff_line):
    base_note = pitch_note(clef)
    return BASE_NOTES[(BASE_NOTES.index(base_note) + staff_line) % 7]

def clef_pitch(clef, staff_line, key=CMAJOR):
    nat_note = clef_note(clef, staff_line)
    note = key[nat_note]
    shift = 0 if note[1] == 'n' else +1 if note[1] == '#' else -1
    return note_pitch(nat_note) + shift
