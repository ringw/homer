import numpy as np

NOTEHEAD = """
   ** 
  ****
 *****
***** 
 ***  """
NOTEHEAD = np.array([c == '*' for c in NOTEHEAD if c != '\n']).reshape((5,6))
BARLINE = np.ones((20, 2), bool)
DOTS = np.zeros((20, 3), bool)
DOTS[6:9] = [[0,1,0],[1,1,1],[0,1,0]]
DOTS[12:15] = [[0,1,0],[1,1,1],[0,1,0]]

def gen_bars(bar_lengths, dots=None):
    pieces = []
    labels = []
    if dots:
        pieces.append(DOTS)
        labels += ['BAR_DOTS' for i in xrange(DOTS.shape[1])]
        space = np.random.randint(1, 6)
        pieces.append(np.zeros((20, space), bool))
        labels += ['BACKGROUND' for i in xrange(pieces[-1].shape[1])]
    for i, length in enumerate(bar_lengths):
        if i > 0:
            space = np.random.randint(1, 6)
            pieces.append(np.zeros((20, space), bool))
            labels += ['BACKGROUND' for i in xrange(pieces[-1].shape[1])]
        pieces.append(np.ones((20, length), bool))
        labels += ['THICK_' * (length > 3) + 'BAR' for k in xrange(length)]
    return np.hstack(pieces), labels

def gen_repeat_left():
    return gen_bars([np.random.randint(1,4), np.random.randint(4,7)], True)
def gen_repeat_right():
    left, labels = gen_repeat_left()
    return left[:,::-1], labels[::-1]
def gen_stemnote(stem=None):
    A = np.zeros((20, 15), bool)
    if stem or (stem is None and np.random.random() < 0.75):
        A[:, 7:9] = 1
    r = np.random.random()
    if r < 0.10:
        A[:int(r*5*20)] = 0
    elif r > 0.90:
        A[int((1-r)*5*20):] = 0
    for i in xrange(np.random.randint(1, 3)):
        side = np.random.random() < 0.5
        x0 = 1 + 7 * side + np.random.randint(0,2)
        y0 = np.random.randint(0, 16)
        A[y0:y0+5, x0:x0+6] = NOTEHEAD
    return A, ['BACKGROUND' for i in xrange(A.shape[1])]
BAR_TYPES = dict(section=lambda: gen_bars([np.random.randint(1,4), np.random.randint(1,4)]),
                 repeat_left=gen_repeat_left,
                 repeat_right=gen_repeat_right,
                 end=lambda: gen_bars([np.random.randint(1,4), np.random.randint(4,7)]),
                 barline=lambda: gen_bars([np.random.randint(1,4)]),
                 stemnote=gen_stemnote)
BAR_PROB = dict(section=0.01, repeat_left=0.02, repeat_right=0.02, barline=0.10, end=0.04, stemnote=0.81)
names = list(BAR_TYPES)
func = [BAR_TYPES[n] for n in names]
pmf = [BAR_PROB[n] for n in names]
cdf = np.cumsum(pmf)

staff = np.zeros(20, bool)
staff[0] = 1
staff[4:6] = 1
staff[9:11] = 1
staff[14:16] = 1
staff[19] = 1
def gen_staff():
    pieces = []
    labels = []
    for i in xrange(np.random.randint(15, 75)):
        k = np.random.random()
        func_ind = (cdf >= k).argmax()
        p,l = func[func_ind]()
        pieces.append(p)
        labels += l
        sp = np.random.randint(3,10)
        pieces.append(np.zeros((20, sp), bool))
        labels += ['BACKGROUND' for s in xrange(sp)]
    pieces = np.hstack(pieces)

    pieces ^= np.random.random(pieces.shape) < 0.02
    pieces[staff, :] ^= np.random.random((staff.sum(), pieces.shape[1])) < 0.10
    return pieces, labels
