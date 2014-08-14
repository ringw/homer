from .gpu import *
from moonshine import forest, components, music
import copy
import music21

CMAJOR = dict((n + 'n', n + 'n') for n in list('abcdefg'))
class PitchState:
    """ Stores clef, key signature, and accidental information """
    BASS = 50 # middle staff line is D3
    TREBLE = 71 # middle staff line is B4
    SHARP = +1
    FLAT = -1
    NATURAL = 0
    clef = None
    piece_key = dict(CMAJOR)
    measure_key = dict(CMAJOR)

def get_musical_elements(page, measure):
    classes = forest.classify(forest.classifier, page, measure.get_image())
    measure.classes_img = classes
    comp_class, comp_bounds, comp_sum = components.get_components(classes)
    comp_class = comp_class.get()
    comp_bounds = comp_bounds.get()
    comp_sum = comp_sum.get()

    staff_dist = 8.0
    component_size = comp_bounds[:,[1,3]] + 1 - comp_bounds[:,[0,2]]
    component_center = (comp_bounds[:,[0,2]] + comp_bounds[:,[1,3]]) / 2.0
    comp_rect_size = np.prod(component_size, axis=1)
    is_blob = ((1 <= component_size) & (component_size < staff_dist)).all(axis=1) & (np.abs(component_size[:,1] - component_size[:,0]) < 42)
    all_elems = np.zeros(classes.shape[1], dtype=np.object)
    for element in ['filled_note', 'empty_note', 'sharp', 'flat', 'natural',
                    'treble_clef', 'bass_clef',
                    'small_treble_clef', 'small_bass_clef']:
        is_class = comp_class == list(forest.classifier.classes).index(element)
        elements = is_class & is_blob #& (#comp_sum >= comp_rect_size*0.25)
            #np.abs(component_size[:,0] - component_size[:,1])
                #< np.max(component_size,axis=1) / 2.0)
        for elem_ind in np.where(elements)[0]:
            x, y = component_center[elem_ind]
            x = np.rint(x).astype(int)
            staff_pos = int(np.rint((measure.staff_y * 8.0/page.staff_dist - y) / 4.0))
            if all_elems[x] != 0:
                all_elems[x].append((element, staff_pos))
            else:
                all_elems[x] = [(element, staff_pos)]
    elem_xs, = np.where(all_elems)
    measure.elements = zip(elem_xs, all_elems[elem_xs])
    #import pylab
    #pylab.figure()
    #pylab.imshow(classes.get())
    return measure.elements

class NotePitch:
    """ Stores a note with pitch information, but no understanding of timing """
    x = None
    filled = None
    staff_line = None
    pitch = None

def parse_notepitches(page, measure, update_keysig=False):
    pitches = []
    keysig = [] if update_keysig else None
    for x, elems in measure.elements:
        for elem, line in elems:
            if elem in ['treble_clef', 'bass_clef']:
                if len(pitches):
                    raise Exception("Clef must be at start of staff")
                if elem == 'treble_clef':
                    if line != -2:
                        print("Treble clef is in an unusual place")
                    measure.pitch.clef = PitchState.TREBLE
                elif elem == 'bass_clef':
                    if line != 2:
                        print("Bass clef is in an unusual place")
                    measure.pitch.clef = PitchState.BASS

            note = music.clef_note(measure.pitch.clef, line)
            if keysig is not None:
                keysig_done = False
                if elem == 'natural':
                    # Key signature doesn't have a natural
                    keysig_done = True
                elif elem == 'sharp':
                    # Each sharp in the key signature is a fifth above
                    # the previous one
                    if keysig and (keysig[-1][1] != '#'
                        or ((ord(keysig[-1][0]) - ord(note[0])) % 7) != 5):
                        keysig_done = True
                    elif keysig == [] and note != 'fn':
                        keysig_done = True
                    else:
                        keysig.append(note[0] + '#')
                elif elem == 'flat':
                    # Each flat in the key signature is a fourth above
                    # the previous one
                    if keysig and (keysig[-1][1] != 'b'
                        or ((ord(keysig[-1][0]) - ord(note[0])) % 7) != 4):
                        keysig_done = True
                    elif keysig == [] and note != 'bn':
                        keysig_done = True
                    else:
                        keysig.append(note[0] + 'b')
                else:
                    # Anything else indicates the end of the key signature
                    keysig_done = True

                if keysig_done:
                    measure.pitch.piece_key.update(
                        (n[0] + 'n', n) for n in keysig)
                    measure.pitch.measure_key = measure.pitch.piece_key
                    keysig = None

            if elem == 'natural':
                measure.pitch.measure_key[note] = note
            elif elem == 'sharp':
                measure.pitch.measure_key[note] = note[0] + '#'
            elif elem == 'flat':
                measure.pitch.measure_key[note] = note[0] + 'b'
            elif elem in ['filled_note', 'empty_note']:
                notepitch = NotePitch()
                notepitch.x = x
                notepitch.filled = elem == 'filled_note'
                notepitch.staff_line = line
                notepitch.pitch = music.clef_pitch(measure.pitch.clef, line,
                                               key=measure.pitch.measure_key)
                pitches.append(notepitch)
    measure.notepitches = pitches
    return pitches

def get_notepitches(page):
    for bar in page.bars:
        # Get each measure from one staff in an array
        for staff in zip(*bar):
            sig = PitchState()
            for m, measure in enumerate(staff):
                get_musical_elements(page, measure)
                measure.pitch = sig
                parse_notepitches(page, measure, update_keysig=(m==0))
                sig = copy.deepcopy(measure.pitch)
                sig.measure_key = sig.piece_key

def get_notepitch_score(page):
    # Parts must be consistent for each system
    assert (np.diff([s['stop'] - s['start'] for s in page.systems]) == 0).all()

    get_notepitches(page)

    doc = music21.stream.Stream()
    for part_num in xrange(page.systems[0]['stop']+1 - page.systems[0]['start']):
        part = music21.stream.Part()
        for bar in page.bars:
            staff = [measure[part_num] for measure in bar]
            for m in staff:
                pitchMeasure = music21.stream.Measure()
                if hasattr(m, 'notepitches'):
                    for notePitch in m.notepitches:
                        note = music21.note.Note(notePitch.pitch)
                        pitchMeasure.append(note)
                part.append(pitchMeasure)
        doc.insert(0, part)
    return doc
