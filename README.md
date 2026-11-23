# Local partitions and fractional packing — computational supplement

This repository accompanies Jinzhou Li’s manuscript *Attaining a fractional packing bound with local partitions: lattice points for entropy, volume for guessing*.

## Start here

For a quick tour, open [the structural-check output](structure_checks.out) and [the prime-field output](twoblock_primes.out). To reproduce them, run the first two commands from the repository root with Python 3. Run the third after installing NumPy to reproduce the exhaustive partition calculations.

```sh
python3 structure_checks.py
python3 twoblock_primes.py
python3 computer_assisted.py
```

Leave Python assertions enabled. The scripts take no input files. The third command writes [the exact spectra](exact_spectra.json) next to the script, so use a working copy if you want to keep the distributed file as a reference. The matching `.out` files record the expected standard output.

| Program | Claim and output |
|---|---|
| `computer_assisted.py` | Appendix B.1–B.2. Bounds the ternary triangle over all partitions, scores the surviving triples exactly and enumerates the binary four-clause network. Writes `exact_spectra.json` beside the script. The ternary entropy minimum in this output is over MAP maximizers. |
| `twoblock_primes.py` | Theorem 6.5 and Appendix B.3. Tests the interval construction through the finite search range and certifies the obstruction inequalities using integer root enclosures. A failed construction test leaves existence at that prime undecided unless the obstruction also applies. |
| `structure_checks.py` | Section 2.6 and Appendix B.4. Verifies the interval-bin counts over the prime field of order 101, quotient and cell examples, the balanced-pair census, two list examples and the slice example. The capped binary counts are tuples of normalized map tables. |

The accompanying `.out` files are the complete standard outputs from Python 3.13.5 with NumPy 2.5.3. The JSON records exact posterior spectra and multiplicities. The displayed floating-point entropies aid interpretation; the entropy ordering of the ternary MAP maximizers uses integer counts and rational products. The scripts include negative controls.
