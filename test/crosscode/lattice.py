"""The cross-code reference ring of test/verify_ring_optics.jl as MAD-X input:
24 FODO cells, 2π of 7.5° sector bends (ρ = 27.5 m), chromatic sextupoles and
one octupole per cell. The same ring is built in Julia by `xring` in that test.
"""
import math

NCELL = 24
ANGLE = 2 * math.pi / (2 * NCELL)
K1F = 1.60; K1D = -1.45
K2F = 6.0; K2D = -9.0
K3 = 400.0


def madx_text(beam):
    return f"""
{beam}
QFH: QUADRUPOLE, L=0.10, K1={K1F};
QD:  QUADRUPOLE, L=0.20, K1={K1D};
SF:  SEXTUPOLE, L=0.05, K2={K2F};
SD:  SEXTUPOLE, L=0.05, K2={K2D};
OC:  OCTUPOLE, L=0.10, K3={K3};
B1:  SBEND, L=3.60, ANGLE={ANGLE!r};
DQS: DRIFT, L=0.05;
DQB: DRIFT, L=0.50;
DQO: DRIFT, L=0.05;
CELL: LINE=(QFH, DQO, OC, DQO, SF, DQB, B1, DQB, SD, DQS, QD, DQS, SD, DQB, B1, DQB, SF, DQS, QFH);
RING: LINE=({NCELL}*CELL);
USE, PERIOD=RING;
"""
