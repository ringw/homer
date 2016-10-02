import env
import metaomr.midi_alignment as ma
from glob import glob
import os.path
import pandas as pd
import cPickle
import re

alignments = dict()
for real in sorted(glob('resources/midi/beet*.mid')):
    f = os.path.basename(real)
    omr = sorted(glob('resources/midi/IMSLP*_' + f))
    if not omr:
        continue
    try:
        real = ma.midi_beats(real)
    except Exception, e:
        print real, '-', e
        continue
    aln = dict()
    for omr_file in omr:
        try:
            omr_beats = ma.midi_beats(omr_file)
        except Exception, e:
            print omr_file, '-', e
            continue
        align = ma.align_measures(real, omr_beats)
        aln[os.path.basename(omr_file)] = align
        print f, os.path.basename(omr_file), align['score'].sum() / float(align['a'][len(align)-1])
    alignments[f] = aln

df = pd.DataFrame(columns='real omr sens spec F1'.split())
i = 0
for real in alignments:
    for omr in alignments[real]:
        align = alignments[real][omr]
        sens = float(align.a_in_b.sum()) / align.nn_b.sum()
        spec = float(align.b_in_a.sum()) / align.nn_a.sum()
        F1 = 2 * sens * spec / ((sens+spec) if sens or spec else 1)
        df.loc[i] = (real, omr, sens, spec, F1)
        i += 1
df.set_index(['real','omr'], inplace=True)

works = cPickle.load(open('imslp/imslp_works.pkl'))
ratings = pd.DataFrame(columns=['rating'])
regex = ''
for title, work in works.iteritems():
    for score in work['scores']:
        if 'rating' in score and score['rating']:
            score_id = os.path.basename(score['public_link'])
            ratings.ix[score_id] = (score['rating'],)

omr_rating = ratings.ix[pd.Series(df.index.get_level_values('omr'), index=df.index).str.extract('IMSLP([0-9]+)')]
df = df.join(omr_rating.set_index(df.index))

df.to_csv('results/beethoven_omr.csv')
