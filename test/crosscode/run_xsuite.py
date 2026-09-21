"""Xsuite reference (pip install xsuite cpymad nafflib): 4d twiss with radiation
integrals, non-linear chromaticity by fit, amplitude detuning by tracking.

    python run_xsuite.py "BEAM, PARTICLE=ELECTRON, ENERGY=3.0;" e 0.51099895e6 3.0e9 linear
    python run_xsuite.py "BEAM, PARTICLE=PROTON, ENERGY=1.938272088;" p 938.27208816e6 1.938272088e9 full

The last argument is the Bend edge model: 'linear' has no dipole fringe
(TrackPad `ExactSBend(...; fringe_bend_entrance=0, fringe_bend_exit=0)`),
'full' includes it (TrackPad `ExactSBend` default; PTC, MAD-X twiss).
Xsuite's ddqx is d2Q/ddelta2 = 2 x TrackPad's chrom2x; det_xx = dQx/dJx with
x = sqrt(2 J beta), the same J as TrackPad's `detuning`.
"""
import sys, json
import xtrack as xt, xpart as xp
from cpymad.madx import Madx
from lattice import madx_text

beam, tag, mass, energy, edge = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
m = Madx(stdout=False); m.input(madx_text(beam))
line = xt.Line.from_madx_sequence(m.sequence.ring)
line.particle_ref = xp.Particles(mass0=mass, energy0=energy)
for el in line.element_dict.values():
    if isinstance(el, (xt.Sextupole, xt.Octupole)):
        el.num_multipole_kicks = 20
    if isinstance(el, xt.Bend):
        el.model = 'rot-kick-rot'; el.edge_entry_model = edge; el.edge_exit_model = edge
line.build_tracker()
tw = line.twiss(method='4d', radiation_integrals=True)
beta0 = float(line.particle_ref.beta0[0]); gamma0 = float(line.particle_ref.gamma0[0])
out = dict(qx=tw.qx, qy=tw.qy, dqx=tw.dqx, dqy=tw.dqy,
           chrom2x_taylor=tw.ddqx / 2, chrom2y_taylor=tw.ddqy / 2,
           alfa=tw.momentum_compaction_factor, slip=tw.slip_factor, length=tw.line_length,
           betx0=tw.betx[0], bety0=tw.bety[0], dx0=tw.dx[0], dpx0=tw.dpx[0],
           I=[tw.rad_int_i1x, tw.rad_int_i2, tw.rad_int_i3, tw.rad_int_i4x, tw.rad_int_i5x],
           beta0=beta0, gamma0=gamma0)
nl = line.get_non_linear_chromaticity(delta0_range=(-2e-3, 2e-3), num_delta=9, fit_order=3)
out.update(fit_dqx=nl.dnqx[1], fit_dqy=nl.dnqy[1], fit_chrom2x_taylor=nl.dnqx[2] / 2, fit_chrom2y_taylor=nl.dnqy[2] / 2)
gemitt = 1e-9   # J1 = 0.5e-9, J2 = 2e-9 m rad, as TrackPad's detuning_actions in the test
det = line.get_amplitude_detuning_coefficients(nemitt_x=gemitt * beta0 * gamma0, nemitt_y=gemitt * beta0 * gamma0,
                                               num_turns=1024, a0_sigmas=0.01, a1_sigmas=1.0, a2_sigmas=2.0)
out.update(det)
print(json.dumps(out, indent=1)); json.dump(out, open(f"xsuite_{tag}_{edge}.json", "w"))
