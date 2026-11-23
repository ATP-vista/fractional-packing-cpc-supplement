"""Computer-assisted statements of the manuscript (Section 8 and Appendix B).

Self-contained; numpy only. Run with Python 3 and NumPy:
    python -u computer_assisted.py > computer_assisted.out

Part T  Triangle over F_3 (Example 2.1, q = 3): every code has P_c <= 108/729 = 4/27 (fibre envelope),
        complete list of optimal deterministic codes, and the minimum of H(T|W) over them,
        3 - (3 + 16 log_3 2)/27.
Part G  Four-clause gadget (Example 8.2): exhaustive F_2 optimum 1/8 (positive control: removing one clause
        node makes 1/4 reachable, and the search finds it); the explicit F_3 code has success 4/81.
Every assertion is preceded by the quantity it tests; failing controls are marked CONTROL.
"""
from itertools import product
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import json
from math import log
import numpy as np

# ---------------------------------------------------------------- Part T: the ternary triangle
# Node u0 holds (a, b), u1 holds (c, d), u2 holds (e, f); targets (a+e, b+c, d+f) mod 3.
# A node map is a partition of the 3x3 grid of its two inputs into at most three fibres
# (output labels do not affect success or entropy).  Cell (x, y) of a grid is bit 3x+y of a mask.

MATS = np.array([[(m >> j) & 1 for j in range(9)] for m in range(512)], dtype=np.int64).reshape(512, 3, 3)
SIZE = MATS.sum((1, 2))


def partitions():
    """Unordered partitions of the 9 cells into <= 3 blocks (restricted growth strings), padded to 3 masks."""
    out = []

    def rec(seq, top):
        if len(seq) == 9:
            out.append(tuple(sum(1 << j for j, x in enumerate(seq) if x == i) for i in range(3)))
            return
        for x in range(min(top + 1, 2) + 1):
            rec(seq + [x], max(top, x))

    rec([0], 0)
    return out


def envelope(largest=True):
    """h[A, t, u] = max over |B| = t, |C| = u of sum_{a,b,d} A(a,b) B(b,d) C(d,a).

    For fixed A and B the best C picks the u largest entries of AB.  largest=False (CONTROL) picks the
    u smallest entries instead, which is not an upper bound."""
    h = np.zeros((512, 10, 10), dtype=np.int64)
    for a in range(512):
        ab = (MATS[a] @ MATS).reshape(512, 9)
        vals = np.sort(ab, axis=1)
        if largest:
            vals = vals[:, ::-1]
        cums = np.concatenate([np.zeros((512, 1), dtype=np.int64), vals.cumsum(1)], axis=1)
        for t in range(10):
            h[a, t] = cums[SIZE == t].max(0)
    return h


def profile(p):
    return tuple(sorted(int(SIZE[m]) for m in p))


def role_scores(HS, t, u):
    """Envelope bound, for every partition at once, for a node whose next two nodes around the triangle have
    fibre-size profiles t and u (see Appendix B for the orientation).  HS[p] = sum of h over the fibres of p."""
    return HS[:, list(t)][:, :, list(u)].sum((1, 2))


RAW = np.array(list(product(range(3), repeat=6)), dtype=np.int64)
A_, B_, C_, D_, E_, F_ = RAW.T
TARGET = 9 * ((A_ + E_) % 3) + 3 * ((B_ + C_) % 3) + (D_ + F_) % 3


def table(p):
    """Map cell index 3x+y -> fibre index."""
    return np.array([next(i for i, m in enumerate(p) if m >> j & 1) for j in range(9)], dtype=np.int64)


def counts(t0, t1, t2):
    z = 9 * t0[3 * A_ + B_] + 3 * t1[3 * C_ + D_] + t2[3 * E_ + F_]
    return np.bincount(27 * z + TARGET, minlength=729).reshape(27, 27)


def score(n):
    return int(n.max(1).sum())


def entropy3(n):
    """H(T|W) in base 3 for the 729 equally likely inputs."""
    tot = n.sum(1)
    hval = 0.0
    for row, s in zip(n, tot):
        for x in row:
            if x:
                hval += x / 729 * log(s / x, 3)
    return hval


def part_T():
    print('Part T: triangle over F_3')
    parts = partitions()
    assert len(parts) == 3281 and len(set(parts)) == 3281
    assert all((a | b | c) == 511 and not (a & b or a & c or b & c) for a, b, c in parts)
    profs = sorted(set(map(profile, parts)))
    by_prof = {s: [p for p in parts if profile(p) == s] for s in profs}
    print(' partitions:', len(parts), ' profiles:', len(profs))
    assert len(profs) == 12
    h = envelope()
    parr = np.array(parts)
    HS = h[parr].sum(1)
    pidx = {p: i for i, p in enumerate(parts)}
    prof_arr = [profile(p) for p in parts]
    in_prof = {s: np.array([i for i, q in enumerate(prof_arr) if q == s]) for s in profs}
    RS = {(t, u): role_scores(HS, t, u) for t in profs for u in profs}
    # U[s,t,u] = max over partitions of profile s of the role score.
    U = {(s, t, u): int(RS[t, u][in_prof[s]].max()) for s in profs for t in profs for u in profs}
    # transpose closure makes U symmetric in its last two arguments
    sym = all(U[s, t, u] == U[s, u, t] for s, t, u in U)
    print(' U symmetric in (t,u):', sym)
    assert sym
    # node u0 sees (u1, u2) in this order, u1 sees (u2, u0), u2 sees (u1, u0)
    bound = {(s, t, u): min(U[s, t, u], U[t, u, s], U[u, t, s]) for s, t, u in product(profs, repeat=3)}
    best = max(bound.values())
    print(' envelope bound on correct inputs out of 729:', best, '->', best, '/ 729')
    assert best == 108
    survivors = [k for k, v in bound.items() if v >= 108]
    print(' ordered profile triples with bound 108:', len(survivors), sorted(set(tuple(sorted(k)) for k in survivors)))
    assert len(survivors) == 7
    assert set(tuple(sorted(k)) for k in survivors) == {((1,4,4),(3,3,3),(3,3,3)), ((2,3,4),(3,3,3),(3,3,3)), ((3,3,3),)*3}
    # Exact classification inside the surviving pools.
    tabs = {}
    optimal = []
    for s, t, u in survivors:
        pool0 = [p for p in by_prof[s] if RS[t, u][pidx[p]] >= 108]
        pool1 = [p for p in by_prof[t] if RS[u, s][pidx[p]] >= 108]
        pool2 = [p for p in by_prof[u] if RS[t, s][pidx[p]] >= 108]
        for p in pool0 + pool1 + pool2:
            if p not in tabs:
                tabs[p] = table(p)
        for p0, p1, p2 in product(pool0, pool1, pool2):
            n = counts(tabs[p0], tabs[p1], tabs[p2])
            sc = score(n)
            assert sc <= 108
            if sc == 108:
                optimal.append((p0, p1, p2, n))
    print(' optimal ordered triples (unlabelled outputs):', len(optimal))
    assert len(optimal) == 4752
    assert len({(a,b,c) for a,b,c,_ in optimal}) == 4752
    ents = Counter()
    hmin, argmin = 9.0, None
    for p0, p1, p2, n in optimal:
        e = round(entropy3(n), 12)
        ents[e] += 1
        if e < hmin:
            hmin, argmin = e, (p0, p1, p2)
    closed = 3 - (3 + 16 * log(2, 3)) / 27
    print(' entropy values among optimal codes (value: count):', dict(sorted(ents.items())))
    print(' minimum H(T|W) among MAP maximizers =', hmin, ' closed form 3-(3+16 log_3 2)/27 =', closed)
    assert abs(hmin - closed) < 1e-9 and hmin > 2.515
    t0, t1, t2 = (tabs[p].reshape(3, 3).tolist() for p in argmin)
    # Exact posterior count spectra (sorted nonzero counts per transcript cell) for each entropy value.
    spectra = {}
    for p0, p1, p2, n in optimal:
        e = round(entropy3(n), 12)
        spec = Counter(tuple(sorted((int(x) for x in row if x), reverse=True)) for row in n)
        spectra.setdefault(e, set()).add(tuple(sorted(spec.items())))
    for e in sorted(spectra):
        print(' spectra at H =', e, ':', spectra[e])
    assert spectra[min(spectra)] == {(((4, 3, 2, 2, 2, 2) + (1,) * 12, 27),)}
    # Certify entropy ordering with integer powers, not rounded logarithms.
    exact_spectra = Counter()
    for pa, pb, pc, n in optimal:
        ta, tb, tc = (table(part).tolist() for part in (pa,pb,pc))
        own = defaultdict(Counter)
        for a,b,c,d,e,f in product(range(3),repeat=6):
            own[ta[3*a+b],tb[3*c+d],tc[3*e+f]][((a+e)%3,(b+c)%3,(d+f)%3)] += 1
        assert sum(max(row.values()) for row in own.values()) == 108
        sp = tuple(sorted(Counter(tuple(sorted(row.values(),reverse=True)) for row in own.values()).items()))
        inherited = tuple(sorted(Counter(tuple(sorted((int(x) for x in row if x),reverse=True)) for row in n).items()))
        assert sp == inherited
        exact_spectra[sp] += 1
    def entropy_product(sp):
        ans = Fraction(1)
        for row, frequency in sp:
            total = sum(row)
            denominator = 1
            for count in row:
                denominator *= count**count
            ans *= Fraction(total**total,denominator)**frequency
        return ans
    ordered = sorted(exact_spectra, key=entropy_product)
    assert len(ordered) == 3
    assert [exact_spectra[sp] for sp in ordered] == [648,216,3888]
    assert ordered[0] == (((4,3,2,2,2,2)+(1,)*12,27),)
    assert entropy_product(ordered[0]) < entropy_product(ordered[1]) < entropy_product(ordered[2])
    Path(__file__).with_name('exact_spectra.json').write_text(json.dumps([
        {'multiplicity':exact_spectra[sp], 'transcript_spectra':sp} for sp in ordered],indent=2)+'\n')
    print(' exact independent spectrum census and rational entropy order: multiplicities [648, 216, 3888], ordered by entropy')
    try:
        assert [exact_spectra[sp] for sp in ordered] == [649,216,3888]
    except AssertionError:
        print(' CONTROL altered entropy census rejected')
    else:
        raise AssertionError('mutated census accepted')
    print(' a minimum-entropy MAP maximizer (rows: first input, columns: second input):', t0, t1, t2)
    # The unbalanced optimal code quoted in Appendix B.
    Ut = np.array([[0, 0, 1], [0, 0, 2], [1, 2, 1]]).ravel()
    Vt = np.array([[0, 0, 1], [0, 2, 2], [1, 2, 1]]).ravel()
    Wt = np.array([[0, 0, 1], [2, 1, 1], [2, 0, 2]]).ravel()
    n = counts(Ut, Vt, Wt)
    print(' unbalanced code U,V,W: correct', score(n), '/ 729, H =', entropy3(n))
    assert score(n) == 108
    balanced = counts(Vt,Vt,Wt)
    assert score(balanced) == 108
    assert Counter(tuple(sorted((int(x) for x in row if x), reverse=True)) for row in balanced) == Counter(dict(ordered[0]))
    # Linear reference: u0 sends b, u1 sends c, u2 sends a constant; the receiver learns b+c only.
    lin = counts(np.array([0, 1, 2] * 3), np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.zeros(9, dtype=np.int64))
    print(' linear one-sum code: correct', score(lin), '/ 729, H =', entropy3(lin))
    assert score(lin) == 81 and abs(entropy3(lin) - 2) < 1e-12
    # CONTROL: an envelope that takes the smallest entries is not a bound; the check must notice.
    hbad = envelope(largest=False)
    p0, p1, p2, _ = optimal[0]
    s, t, u = profile(p0), profile(p1), profile(p2)
    bad = int(role_scores(hbad[np.array([p0])].sum(1), t, u)[0])
    print(' CONTROL smallest-entry envelope for an optimal code:', bad, '(< 108 so it would be refuted)')
    assert bad < 108
    return optimal


# ---------------------------------------------------------------- Part G: the four-clause gadget
DEM = ['AT', 'AF', 'BT', 'BF']
NODES = [('AT', 'AF'), ('BT', 'BF'), ('AF', 'BF'), ('AF', 'BT'), ('AT', 'BF'), ('AT', 'BT')]


def exhaustive_f2(nodes):
    """Optimal success over F_2 of all deterministic codes; one output bit per node, full inputs."""
    inc = [(w, d) for w, pair in enumerate(nodes) for d in pair]
    N = 1 << len(inc)
    x = (np.arange(N)[:, None] >> np.arange(len(inc))[None, :]) & 1
    tgt = np.zeros(N, dtype=np.int64)
    for k, d in enumerate(DEM):
        s = np.zeros(N, dtype=np.int64)
        for j, (w, dd) in enumerate(inc):
            if dd == d:
                s ^= x[:, j]
        tgt |= s << k
    win = [2 * x[:, 2 * w] + x[:, 2 * w + 1] for w in range(len(nodes))]
    maps = [m for m in range(16) if not m & 1]  # f(0) = 0: output complement does not change success
    bits = [[(m >> win[w]) & 1 for m in maps] for w in range(len(nodes))]
    k = len(nodes)
    head, tail = k // 2, k - k // 2
    inner = []
    for ms in product(range(8), repeat=tail):
        v = np.zeros(N, dtype=np.int64)
        for j, m in enumerate(ms):
            v += bits[head + j][m] << (head + j)
        inner.append(v)
    inner = np.array(inner)
    R = inner.shape[0]
    best = 0
    for ms in product(range(8), repeat=head):
        v = np.zeros(N, dtype=np.int64)
        for j, m in enumerate(ms):
            v += bits[j][m] << j
        key = ((v[None, :] + inner) << 4) + tgt[None, :] + (np.arange(R)[:, None] << (k + 4))
        c = np.bincount(key.ravel(), minlength=R << (k + 4)).reshape(R, 1 << k, 16)
        best = max(best, int(c.max(2).sum(1).max()))
    return best, N


GADGET_F3 = [  # table[w][first input][second input], node inputs as in NODES
    [[2, 0, 2], [1, 1, 2], [1, 0, 0]],
    [[1, 1, 1], [2, 2, 2], [0, 0, 0]],
    [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
    [[2, 0, 0], [2, 2, 1], [1, 0, 1]],
    [[2, 2, 2], [1, 1, 1], [0, 0, 0]],
    [[1, 0, 0], [2, 1, 2], [2, 0, 0]],
]


def gadget_f3():
    inc = [(w, d) for w, pair in enumerate(NODES) for d in pair]
    raw = np.array(list(product(range(3), repeat=12)), dtype=np.int64)
    tgt = np.zeros(len(raw), dtype=np.int64)
    for k, d in enumerate(DEM):
        s = sum(raw[:, j] for j, (w, dd) in enumerate(inc) if dd == d) % 3
        tgt += s * 3 ** k
    z = np.zeros(len(raw), dtype=np.int64)
    for w in range(6):
        tw = np.array(GADGET_F3[w])
        z += tw[raw[:, 2 * w], raw[:, 2 * w + 1]] * 3 ** w
    c = np.bincount(z * 81 + tgt, minlength=729 * 81).reshape(729, 81)
    return int(c.max(1).sum()), len(raw)


def part_G():
    print('Part G: four-clause gadget')
    best, N = exhaustive_f2(NODES)
    print(' F_2 exhaustive optimum:', best, '/', N, '=', best / N)
    assert best * 8 == N
    best5, N5 = exhaustive_f2(NODES[:-1])
    print(' CONTROL without clause node {AT,BT}: optimum', best5, '/', N5, '=', best5 / N5, '(bound 1/4 now reachable)')
    assert best5 * 4 == N5
    good, N3 = gadget_f3()
    print(' F_3 explicit code: correct', good, '/', N3, '=', good / N3, ' (4/81 =', 4 / 81, ')')
    assert good * 81 == 4 * N3


if __name__ == '__main__':
    part_T()
    part_G()
    print('ALL ASSERTIONS PASSED')
