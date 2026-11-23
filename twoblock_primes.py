"""Supplementary computation for the two-block theorem (label thm:twoblock).

(A) For each prime p <= 3000, search integers s1, s2 >= 1 with
      ceil(p/s1)*ceil(p/s2) <= p  and  floor(p/(3 s_j - 2))**3 >= p  (j = 1, 2),
    the parameter conditions of the interval-box construction (eq. twoblock-params).
    Report the primes >= 911 where no parameters exist and the smallest prime with parameters.
(B) Check the small-prime obstruction inequalities with exact rational enclosures:
    case "one full sumset":  (6p/(sqrt p + 2))**3 > p**2 * (1 + 1/(2 sqrt p - 1))**3   for 5 <= p <= 619,
                              (2 sqrt p - 1)**3 > p**2                                 for p = 2, 3;
    case "no full sumset":   p*(1 + 2 p**(-1/3))**6 < 729                              for p <= 61,
                              3 sqrt p > p**(2/3) + 2                                   for 67 <= p <= 619.
    Controls: the first failing prime of each inequality is printed, so the assertions can be seen to fail
    just outside the claimed ranges.
"""
from fractions import Fraction as Q
from functools import lru_cache
from math import isqrt

SCALE = 10**12

def icbrt(n):
    lo, hi = 0, 1 << ((n.bit_length()+2)//3)
    while lo+1 < hi:
        mid=(lo+hi)//2
        if mid**3 <= n: lo=mid
        else: hi=mid
    assert lo**3 <= n < (lo+1)**3
    return lo

@lru_cache(None)
def enclosures(p):
    s=isqrt(p*SCALE**2)
    t=icbrt(p*SCALE**3)
    u=icbrt(p*p*SCALE**3)
    return Q(s,SCALE),Q(s+1,SCALE),Q(t,SCALE),Q(t+1,SCALE),Q(u,SCALE),Q(u+1,SCALE)


def primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    return [i for i in range(n + 1) if sieve[i]]


def ceil_div(a, b):
    return -(-a // b)


def has_params(p):
    good = [s for s in range(1, p + 1) if (p // (3 * s - 2)) ** 3 >= p]
    if not good:
        return None
    # ceil(p/s) is nonincreasing in s, so the largest admissible s is best for each factor
    s = max(good)
    if ceil_div(p, s) * ceil_div(p, s) <= p:
        return (s, s)
    return None


P = primes_upto(3000)
ok = {p: has_params(p) for p in P}
smallest = min(p for p in P if ok[p])
missing = [p for p in P if p >= 911 and not ok[p]]
print("smallest prime with parameters:", smallest, ok[smallest])
print("primes in [911,3000] without parameters:", missing)
assert smallest == 911
assert missing == [937] + [p for p in P if 1009 <= p <= 1097] + [1123, 1129], missing
assert all(ok[p] for p in P if 1130 <= p <= 3000)

# (B) Every positive/negative decision below is certified with rational bounds.
def case_one(p):
    sl,su,cl,cu,ul,uu=enclosures(p)
    lower=6*p/(su+2)-uu*(1+1/(2*sl-1))
    upper=6*p/(sl+2)-ul*(1+1/(2*su-1))
    assert lower>0 or upper<0, ('inconclusive enclosure',p)
    return lower>0


def case_none_small(p):
    sl,su,cl,cu,ul,uu=enclosures(p)
    lower=p*(1+2/cu)**6
    upper=p*(1+2/cl)**6
    assert upper<729 or lower>729, ('inconclusive enclosure',p)
    return upper<729


def case_none_sharp(p):
    sl,su,cl,cu,ul,uu=enclosures(p)
    lower=3*sl-uu-2
    upper=3*su-ul-2
    assert lower>0 or upper<0, ('inconclusive enclosure',p)
    return lower>0


assert all((2*enclosures(p)[0]-1)**3>p*p for p in (2,3))
assert all(case_one(p) for p in P if 5<=p<=619)
assert all(case_none_small(p) for p in P if p<=61)
assert all(case_none_sharp(p) for p in P if 67<=p<=619)
sharp_failure=min(p for p in P if p>619 and not case_none_sharp(p))
small_failure=min(p for p in P if not case_none_small(p))
one_failure=min(p for p in primes_upto(200000) if p>5 and not case_one(p))
assert (sharp_failure,small_failure,one_failure)==(631,67,43427)
print('exact rational enclosures: scale =', SCALE)
print('first prime > 619 where the sharp no-full-sumset inequality fails:',sharp_failure)
print('first prime where the small-p no-full-sumset inequality fails:',small_failure)
print('first prime > 5 where the one-full-sumset inequality fails:',one_failure)
assert 1883**3>2025*142**3
assert 21*45**2+98*45-1519>0
print('infinite-tail analytic endpoint and derivative certificates: exact integer assertions passed')
print('all assertions passed')
