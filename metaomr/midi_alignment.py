import numpy as np
import pandas as pd
import skimage.measure
import scipy.spatial.distance as ssd
from metaomr import bitimage
from glob import glob
import os.path
import scipy.misc
from music21 import converter, meter
from collections import defaultdict

RESOLUTION = (2 * 3) ** 6 * (5 * 7)
def midi_beats(midi_file):
    music = converter.parse(midi_file)
    # Use dummy 1/4 time
    for part in music.parts:
        for sig in part.getTimeSignatures():
            denom = max(4, sig.denominator)
            part.insert(sig.offset, meter.TimeSignature('1/%d' % denom))
            part.remove(sig)
    measures = []
    for measure in music.flat.makeMeasures():
        offsetMap = defaultdict(set)
        for note in measure.notes:
            offset = int(round(note.offset * RESOLUTION))
            offsetMap[offset].update([pitch.midi for pitch in note.pitches])
        measures.append(offsetMap)
    return measures

def measure_beat_dists(ma, mb):
    scores = np.empty((len(ma), len(mb)), float)
    notes_a = np.array([sum(map(len, ma[i].values())) for i in xrange(len(ma))])
    notes_b = np.array([sum(map(len, mb[j].values())) for j in xrange(len(mb))])
    for i in xrange(len(ma)):
        for j in xrange(len(mb)):
            numnotes_a = notes_a[i]
            numnotes_b = notes_b[j]
            a_in_b = 0
            for offset in ma[i]:
                if offset in mb[j]:
                    a_in_b += len(ma[i][offset].intersection(mb[j][offset]))
            b_in_a = 0
            for offset in mb[j]:
                if offset in ma[i]:
                    b_in_a += len(mb[j][offset].intersection(ma[i][offset]))
            if numnotes_a and numnotes_b:
                sens = float(a_in_b) / numnotes_a
                spec = float(b_in_a) / numnotes_b
                F1 = (2 * sens * spec / (sens + spec)
                      if sens or spec else 0.0)
                scores[i, j] = F1
            elif numnotes_a == numnotes_b == 0:
                scores[i, j] = 1
            else:
                scores[i, j] = 0
    dists = 1 - scores
    dists *= np.maximum(notes_a[:, None], notes_b[None, :])
    return dists

def align_measures(ma, mb, gap_penalty=10):
    dists = measure_beat_dists(ma, mb)
    scores = np.empty((len(ma), len(mb)))
    scores[0, 0] = dists[0, 0]
    for i in xrange(len(ma)):
        scores[i, 0] = i * gap_penalty
    for j in xrange(len(mb)):
        scores[0, j] = j * gap_penalty
    dx = np.array([-1, -1, 0], int)
    dy = np.array([-1, 0, -1], int)
    ptr = np.empty_like(scores, int)
    ptr[0, 0] = 0
    ptr[1:, 0] = 2
    ptr[0, 1:] = 1
    for i in xrange(1, len(ma)):
        for j in xrange(1, len(mb)):
            new_scores = scores[i + dy, j + dx]
            new_scores[0] += dists[i, j]
            new_scores[1:] += gap_penalty
            ptr[i, j] = np.argmin(new_scores)
            scores[i, j] = new_scores[ptr[i, j]]
    score = scores[i, j]
    alignment = []
    while i >= 0 and j >= 0:
        direction = ptr[i, j]
        alignment.append((i if direction != 1 else -1,
                          j if direction != 2 else -1,
                          dists[i, j] if direction == 0 else gap_penalty))
        i += dy[direction]
        j += dx[direction]
    alignment = alignment[::-1]
    alignment = pd.DataFrame(alignment, columns='a b score'.split())

    notes_a = np.array([sum(map(len, ma[i].values())) for i in xrange(len(ma))])
    notes_b = np.array([sum(map(len, mb[j].values())) for j in xrange(len(mb))])
    alignment['nn_a'] = notes_a[np.array(alignment['a'], int)]
    alignment['nn_b'] = notes_b[np.array(alignment['b'], int)]
    a_in_b = [sum([len(ma[i][offset].intersection(mb[j][offset]))
                   if offset in mb[j] else 0
                   for offset in ma[i]])
              for i, j in np.array(alignment[['a', 'b']])]
    b_in_a = [sum([len(mb[j][offset].intersection(ma[i][offset]))
                   if offset in ma[i] else 0
                   for offset in mb[j]])
              for i, j in np.array(alignment[['a', 'b']])]
    alignment['a_in_b'] = a_in_b
    alignment['b_in_a'] = b_in_a
    return alignment
