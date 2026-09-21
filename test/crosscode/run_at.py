"""pyAT reference (pip install accelerator-toolbox): the AT expanded bend that
TrackPad's `SBend` reproduces bit for bit, and AT's exact sector bend.
AT's PolynomB is k_n/n!, hence the /2 and /6.
"""
import numpy as np
import at
from lattice import NCELL, ANGLE, K1F, K1D, K2F, K2D, K3


def mk(pm, n=20):
    QFH = at.Quadrupole('QFH', 0.10, K1F, NumIntSteps=n); QD = at.Quadrupole('QD', 0.20, K1D, NumIntSteps=n)
    SF = at.Sextupole('SF', 0.05, K2F / 2, NumIntSteps=n); SD = at.Sextupole('SD', 0.05, K2D / 2, NumIntSteps=n)
    OC = at.Octupole('OC', 0.10, [0, 0, 0, 0], [0, 0, 0, K3 / 6], NumIntSteps=n)
    B1 = at.Dipole('B1', 3.60, ANGLE, NumIntSteps=n, PassMethod=pm)
    DQS = at.Drift('DQS', 0.05); DQB = at.Drift('DQB', 0.50); DQO = at.Drift('DQO', 0.05)
    cell = [QFH, DQO, OC, DQO, SF, DQB, B1, DQB, SD, DQS, QD, DQS, SD, DQB, B1, DQB, SF, DQS, QFH]
    return at.Lattice(cell * NCELL, energy=3e9, periodicity=1)


for pm in ['BndMPoleSymplectic4Pass', 'ExactSectorBendPass']:
    ring = mk(pm)
    _, rd, _ = at.linopt6(ring, get_chrom=True)
    print(pm, 'tune', rd.tune, 'chrom', rd.chromaticity)
    ds = np.linspace(-2e-3, 2e-3, 9); qs = []
    for d in ds:
        _, r, _ = at.linopt6(ring, dp=d); qs.append(r.tune[:2])
    qs = np.array(qs); cx = np.polyfit(ds, qs[:, 0], 3); cy = np.polyfit(ds, qs[:, 1], 3)
    print('   fit chrom', cx[2], cy[2], ' Taylor chrom2', cx[1], cy[1])
    print('   I1..I5', at.get_radiation_integrals(ring))
