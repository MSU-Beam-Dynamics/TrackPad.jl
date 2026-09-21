module FODOCellWithDipolesExample

using LinearAlgebra
using TrackPad

export build_fodo_cell, run_example, sample_fodo_optics

"""
    build_fodo_cell(; start=:mid_qf, kwargs...)

Build one periodic electron FODO cell with finite-length quadrupoles and two
sector dipoles. `start` may be `:mid_qf`, `:mid_qd`, `:after_qf`, or
`:after_qd`; these are cyclic representations of the same physical cell.
"""
function build_fodo_cell(; start::Symbol = :mid_qf,
                         half_cell_length::Float64 = 5.0,
                         phase_advance::Float64 = pi / 2,
                         quadrupole_length::Float64 = 0.02,
                         bend_angle::Float64 = 0.03,
                         integration_steps::Int = 40)
    0 < phase_advance < pi || throw(ArgumentError("phase_advance must be in (0, pi)"))
    0 < quadrupole_length < half_cell_length ||
        throw(ArgumentError("quadrupole_length must be in (0, half_cell_length)"))

    s = sin(phase_advance / 2)
    focal_length = half_cell_length / (2s)
    k1 = inv(focal_length * quadrupole_length)
    bend_length = half_cell_length - quadrupole_length

    qf(L, name) = Quadrupole(L, k1; name, num_int_steps = integration_steps)
    qd(L, name) = Quadrupole(L, -k1; name, num_int_steps = integration_steps)
    bend(name) = SBend(
        bend_length, bend_angle;
        name,
        num_int_steps = integration_steps,
        fringe_bend_entrance = 0,
        fringe_bend_exit = 0,
    )

    elements = if start === :mid_qf
        AbstractElement[
            qf(quadrupole_length / 2, :QF_HALF_1),
            bend(:BEND_FD),
            qd(quadrupole_length, :QD),
            bend(:BEND_DF),
            qf(quadrupole_length / 2, :QF_HALF_2),
        ]
    elseif start === :mid_qd
        AbstractElement[
            qd(quadrupole_length / 2, :QD_HALF_1),
            bend(:BEND_DF),
            qf(quadrupole_length, :QF),
            bend(:BEND_FD),
            qd(quadrupole_length / 2, :QD_HALF_2),
        ]
    elseif start === :after_qf
        AbstractElement[
            bend(:BEND_FD),
            qd(quadrupole_length, :QD),
            bend(:BEND_DF),
            qf(quadrupole_length, :QF),
        ]
    elseif start === :after_qd
        AbstractElement[
            bend(:BEND_DF),
            qf(quadrupole_length, :QF),
            bend(:BEND_FD),
            qd(quadrupole_length, :QD),
        ]
    else
        throw(ArgumentError("start must be :mid_qf, :mid_qd, :after_qf, or :after_qd"))
    end

    return Lattice(elements; name = :FODO_WITH_DIPOLES, periodic = true)
end

function thin_lens_matrices(L1, f)
    common = 1 - L1^2 / (2f^2)
    mid_qf = [
        common                         L1 * (L1 + 2f) / f
        L1 * (L1 - 2f) / (4f^3)       common
    ]
    mid_qd = [
        common                         L1 * (-L1 + 2f) / f
        -L1 * (L1 + 2f) / (4f^3)      common
    ]
    # This is the physical map immediately after QF for the signs in Eq. 9.1.
    after_qf = [
        1 + L1 / f                     L1 * (L1 + 2f) / f
        -L1 / f^2                      1 - L1 / f - L1^2 / f^2
    ]
    # Equations 9.4 and 9.9 call this boundary "after QF", but their matrix is
    # the physical after-QD cyclic map under the focusing signs of Eq. 9.1.
    note_after_qf = [
        1 - L1 / f                     L1 * (-L1 + 2f) / f
        -L1 / f^2                      1 + L1 / f - L1^2 / f^2
    ]
    return (; mid_qf, mid_qd, after_qf, note_after_qf)
end

function thin_dispersion_matrix(L1, f, theta)
    return [
        1 - L1^2 / (2f^2)             L1 * (L1 + 2f) / f             L1 * theta * (L1 + 4f) / (2f)
        L1 * (L1 - 2f) / (4f^3)       1 - L1^2 / (2f^2)             theta * (-L1^2 - 2L1 * f + 8f^2) / (4f^2)
        0.0                            0.0                            1.0
    ]
end

function plane_optics(M::AbstractMatrix, i::Int)
    A = Matrix(@view M[i:(i + 1), i:(i + 1)])
    cosmu = clamp(tr(A) / 2, -1.0, 1.0)
    mu = acos(cosmu)
    sinmu = sin(mu)
    if A[1, 2] / sinmu < 0
        mu = 2pi - mu
        sinmu = sin(mu)
    end
    beta = A[1, 2] / sinmu
    alpha = (A[1, 1] - A[2, 2]) / (2sinmu)
    return (; A, mu, beta, alpha, gamma = (1 + alpha^2) / beta)
end

function periodic_dispersion(M::AbstractMatrix, beam::Beam)
    A = Matrix(@view M[1:2, 1:2])
    forcing = Vector(@view M[1:2, 6])
    dispersion_energy = (I - A) \ forcing

    # TrackPad's sixth coordinate is delta_E. The note differentiates with
    # respect to delta_P, and delta_E = beta0 * delta_P to first order.
    return beam.beta .* dispersion_energy
end

h_function(optics, dispersion) =
    optics.gamma * dispersion[1]^2 +
    2optics.alpha * dispersion[1] * dispersion[2] +
    optics.beta * dispersion[2]^2

function analytic_quantities(L1, f, theta)
    s = L1 / (2f)
    c = sqrt(1 - s^2)
    mu = 2asin(s)
    beta_f = 2L1 * (1 + s) / sin(mu)
    beta_d = 2L1 * (1 - s) / sin(mu)
    dispersion_f = L1 * theta / (2s) * (1 + 2 / s)
    dispersion_d = L1 * theta / (2s) * (-1 + 2 / s)
    h_f = L1 * theta^2 * c * (1 + s / 2)^2 / (s^3 * (1 + s))
    h_d = L1 * theta^2 * c * (1 - s / 2)^2 / (s^3 * (1 - s))
    chromaticity = -tan(mu / 2) / pi
    return (;
        s, mu, beta_f, beta_d, dispersion_f, dispersion_d,
        h_f, h_d, h_average = (h_f + h_d) / 2, chromaticity,
    )
end

function phase_scan(; points::Int = 100, epsilon::Float64 = 0.2)
    phase = collect(range(epsilon, pi - epsilon; length = points))
    s = sin.(phase ./ 2)
    beta_max = 2 .* (1 .+ s) ./ sin.(phase)
    beta_min = 2 .* (1 .- s) ./ sin.(phase)
    dispersion_max = (s .+ 2) ./ (2 .* s.^2)
    dispersion_min = (-s .+ 2) ./ (2 .* s.^2)
    h_f = dispersion_max.^2 ./ beta_max
    h_d = dispersion_min.^2 ./ beta_min
    h_average = (h_f .+ h_d) ./ 2
    specific_chromaticity = -tan.(phase ./ 2) ./ (phase ./ 2)
    return (;
        phase, beta_max, beta_min, dispersion_max, dispersion_min,
        h_f, h_d, h_average, specific_chromaticity,
    )
end

function h_optimum()
    # Continuous minimum of <H>/(L1*theta^2).
    s2 = (19 - sqrt(73)) / 12
    s = sqrt(s2)
    mu = 2asin(s)
    normalized_h = (1 - 3s2 / 4) / (s^3 * sqrt(1 - s2))

    # Reproduce the published notebook's 100-point plotting-grid minimum.
    scan = phase_scan()
    grid_index = argmin(scan.h_average)
    grid_mu = scan.phase[grid_index]
    return (;
        mu,
        degrees = rad2deg(mu),
        normalized_h,
        grid_mu,
        grid_degrees = rad2deg(grid_mu),
        grid_normalized_h = scan.h_average[grid_index],
    )
end

function require_close(label, actual, expected; rtol = 0.0, atol = 0.0)
    isapprox(actual, expected; rtol, atol) && return nothing
    throw(AssertionError(
        "$label: actual=$actual, expected=$expected, rtol=$rtol, atol=$atol",
    ))
end

function relative_error(actual, expected)
    return maximum(abs.(actual .- expected)) / max(maximum(abs.(expected)), eps(Float64))
end

function _slice_fodo_element(elem::Quadrupole, slices::Int)
    steps = max(1, cld(elem.num_int_steps, slices))
    return AbstractElement[
        Quadrupole(elem.L / slices, elem.k1; name = elem.name, num_int_steps = steps)
        for _ in 1:slices
    ]
end

function _slice_fodo_element(elem::SBend, slices::Int)
    steps = max(1, cld(elem.num_int_steps, slices))
    return AbstractElement[
        SBend(
            elem.L / slices, elem.angle / slices;
            name = elem.name,
            num_int_steps = steps,
            fringe_bend_entrance = 0,
            fringe_bend_exit = 0,
        )
        for _ in 1:slices
    ]
end

"""
    sample_fodo_optics(; quadrupole_slices=8, bend_slices=40)

Return smoothly sampled periodic beta functions and horizontal dispersion for
the finite-magnet FODO example. Magnets are sliced only to provide plotting
reference points; their total lengths, strengths, and bend angles are
unchanged.
"""
function sample_fodo_optics(; quadrupole_slices::Int = 8,
                            bend_slices::Int = 40)
    quadrupole_slices > 0 || throw(ArgumentError("quadrupole_slices must be positive"))
    bend_slices > 0 || throw(ArgumentError("bend_slices must be positive"))

    beam = Beam(3.0e9)
    cell = build_fodo_cell(; start = :mid_qf)
    slices = AbstractElement[]
    for elem in cell
        count = elem isa Quadrupole ?
            max(1, round(Int, quadrupole_slices * elem.L / 0.02)) :
            bend_slices
        append!(slices, _slice_fodo_element(elem, count))
    end
    sampled_cell = Lattice(slices; name = :SAMPLED_FODO_WITH_DIPOLES, periodic = true)
    twiss = twissline(sampled_cell, beam)

    segment_maps = fastfindm66_refpts(
        sampled_cell, 0.0, collect(1:length(sampled_cell));
        E0 = beam.energy, m0 = beam.mass,
    )
    full_map = one_turn_map(sampled_cell, beam)
    dispersion_energy = (I - full_map[1:2, 1:2]) \ full_map[1:2, 6]
    dispersion = zeros(length(sampled_cell) + 1)
    dispersion[1] = beam.beta * dispersion_energy[1]
    for i in 1:length(sampled_cell)
        segment = @view segment_maps[:, :, i]
        dispersion_energy = segment[1:2, 1:2] * dispersion_energy + segment[1:2, 6]
        dispersion[i + 1] = beam.beta * dispersion_energy[1]
    end

    return (;
        lattice = sampled_cell,
        s = twiss.s,
        betax = twiss.betax,
        betay = twiss.betay,
        dispersion,
        tunex = twiss.tunex,
        tuney = twiss.tuney,
    )
end

"""
    run_example(; check=true, verbose=true)

Run the complete FODO-cell demonstration and return all calculated values.
With `check=true`, verify the matrix, stability, Twiss, dispersion,
chromaticity, radiation, and phase-optimization results.
"""
function run_example(; check::Bool = true, verbose::Bool = true)
    L1 = 5.0
    requested_mu = pi / 2
    theta = 0.03
    Lq = 0.02
    beam = Beam(3.0e9)
    focal_length = L1 / (2sin(requested_mu / 2))

    lattices = (;
        mid_qf = build_fodo_cell(
            start = :mid_qf, half_cell_length = L1,
            phase_advance = requested_mu, quadrupole_length = Lq,
            bend_angle = theta,
        ),
        mid_qd = build_fodo_cell(
            start = :mid_qd, half_cell_length = L1,
            phase_advance = requested_mu, quadrupole_length = Lq,
            bend_angle = theta,
        ),
        after_qf = build_fodo_cell(
            start = :after_qf, half_cell_length = L1,
            phase_advance = requested_mu, quadrupole_length = Lq,
            bend_angle = theta,
        ),
        after_qd = build_fodo_cell(
            start = :after_qd, half_cell_length = L1,
            phase_advance = requested_mu, quadrupole_length = Lq,
            bend_angle = theta,
        ),
    )

    maps = (;
        mid_qf = one_turn_map(lattices.mid_qf, beam),
        mid_qd = one_turn_map(lattices.mid_qd, beam),
        after_qf = one_turn_map(lattices.after_qf, beam),
        after_qd = one_turn_map(lattices.after_qd, beam),
    )
    thin = thin_lens_matrices(L1, focal_length)
    thin_dispersion = thin_dispersion_matrix(L1, focal_length, theta)
    analytic = analytic_quantities(L1, focal_length, theta)

    optics = (;
        mid_qf = plane_optics(maps.mid_qf, 1),
        mid_qd = plane_optics(maps.mid_qd, 1),
        vertical = plane_optics(maps.mid_qf, 3),
    )
    dispersion = (;
        mid_qf = periodic_dispersion(maps.mid_qf, beam),
        mid_qd = periodic_dispersion(maps.mid_qd, beam),
    )
    h = (;
        mid_qf = h_function(optics.mid_qf, dispersion.mid_qf),
        mid_qd = h_function(optics.mid_qd, dispersion.mid_qd),
    )

    # getchrom defaults to the note's momentum convention, dQ/d(delta_P).
    chrom_momentum = getchrom(
        lattices.mid_qf, beam;
        centered = true, dpp = 1.0e-5,
    )
    chrom_energy = getchrom(
        lattices.mid_qf, beam;
        centered = true, dpp = 1.0e-5, wrt = :deltae,
    )

    bend_length = L1 - Lq
    rho = bend_length / theta
    i2 = 2bend_length / rho^2
    h_average = (h.mid_qf + h.mid_qd) / 2
    i5_endpoint = 2bend_length * h_average / rho^3
    radiation = (; rho, i2, i5_endpoint, rho_i5_over_i2 = rho * i5_endpoint / i2)
    scan = phase_scan()
    optimum = h_optimum()

    errors = (;
        matrix_mid_qf = relative_error(optics.mid_qf.A, thin.mid_qf),
        matrix_mid_qd = relative_error(optics.mid_qd.A, thin.mid_qd),
        matrix_after_qf = relative_error(maps.after_qf[1:2, 1:2], thin.after_qf),
        matrix_note_after_qf = relative_error(maps.after_qd[1:2, 1:2], thin.note_after_qf),
        beta_f = abs(optics.mid_qf.beta / analytic.beta_f - 1),
        beta_d = abs(optics.mid_qd.beta / analytic.beta_d - 1),
        dispersion_f = abs(dispersion.mid_qf[1] / analytic.dispersion_f - 1),
        dispersion_d = abs(dispersion.mid_qd[1] / analytic.dispersion_d - 1),
        chromaticity_x = abs(chrom_momentum[1] / analytic.chromaticity - 1),
        h_f = abs(h.mid_qf / analytic.h_f - 1),
        h_d = abs(h.mid_qd / analytic.h_d - 1),
    )

    if check
        for (name, matrix) in pairs(thin)
            require_close("thin-lens determinant ($name)", det(matrix), 1.0; atol = 2e-15)
            require_close("thin-lens trace ($name)", tr(matrix), 2cos(analytic.mu); atol = 2e-15)
        end
        require_close("thin-lens phase advance", analytic.mu, requested_mu; atol = 2e-15)
        L1 < 2focal_length || throw(AssertionError("FODO stability condition L1 < 2f failed"))

        require_close("real-map determinant at QF", det(optics.mid_qf.A), 1.0; atol = 2e-8)
        require_close("real-map determinant at QD", det(optics.mid_qd.A), 1.0; atol = 2e-8)
        require_close("cyclic trace at QD", tr(optics.mid_qd.A), tr(optics.mid_qf.A); atol = 2e-8)
        require_close(
            "cyclic trace after QF", tr(maps.after_qf[1:2, 1:2]), tr(optics.mid_qf.A);
            atol = 2e-8,
        )
        require_close(
            "cyclic trace after QD", tr(maps.after_qd[1:2, 1:2]), tr(optics.mid_qf.A);
            atol = 2e-8,
        )

        errors.matrix_mid_qf < 0.015 || throw(AssertionError("mid-QF thin-lens matrix error is too large"))
        errors.matrix_mid_qd < 0.015 || throw(AssertionError("mid-QD thin-lens matrix error is too large"))
        errors.matrix_after_qf < 0.015 || throw(AssertionError("after-QF thin-lens matrix error is too large"))
        errors.matrix_note_after_qf < 0.015 || throw(AssertionError("note Eq. 9.4 matrix error is too large"))
        require_close("extended-map transverse block", thin_dispersion[1:2, 1:2], thin.mid_qf; atol = 2e-15)
        thin_periodic_dispersion = (I - thin_dispersion[1:2, 1:2]) \ thin_dispersion[1:2, 3]
        require_close(
            "thin periodic dispersion at QF", thin_periodic_dispersion,
            [analytic.dispersion_f, 0.0]; atol = 3e-16,
        )
        errors.beta_f < 0.015 || throw(AssertionError("beta maximum differs from thin-lens result"))
        errors.beta_d < 0.015 || throw(AssertionError("beta minimum differs from thin-lens result"))
        abs(optics.mid_qf.alpha) < 2e-3 || throw(AssertionError("alpha at QF center is not zero"))
        abs(optics.mid_qd.alpha) < 2e-3 || throw(AssertionError("alpha at QD center is not zero"))
        errors.dispersion_f < 0.025 || throw(AssertionError("QF dispersion differs from small-angle result"))
        errors.dispersion_d < 0.025 || throw(AssertionError("QD dispersion differs from small-angle result"))
        abs(dispersion.mid_qf[2]) < 2e-5 || throw(AssertionError("D' at QF center is not zero"))
        abs(dispersion.mid_qd[2]) < 2e-5 || throw(AssertionError("D' at QD center is not zero"))
        errors.chromaticity_x < 0.035 || throw(AssertionError("natural chromaticity differs from thin-lens result"))
        errors.h_f < 0.04 || throw(AssertionError("H at QF differs from endpoint formula"))
        errors.h_d < 0.04 || throw(AssertionError("H at QD differs from endpoint formula"))
        require_close("endpoint radiation estimate", radiation.rho_i5_over_i2, h_average; atol = 2e-15)
        require_close("notebook-grid phase advance", optimum.grid_degrees, 138.39385343374136; atol = 2e-12)
        require_close("notebook-grid normalized H", optimum.grid_normalized_h, 1.18777034003462; atol = 2e-14)
        require_close("continuous optimal phase advance", optimum.degrees, 137.95901942343284; atol = 2e-12)
        require_close("continuous minimum normalized H", optimum.normalized_h, 1.187664351886066; atol = 2e-14)
        optimum.normalized_h < optimum.grid_normalized_h ||
            throw(AssertionError("continuous H minimum must improve on the plotting grid"))
    end

    if verbose
        println("FODO cell with finite quadrupoles and sector dipoles")
        println("  beam total energy [GeV]         = ", beam.energy / 1e9)
        println("  half-cell length L1 [m]         = ", L1)
        println("  quadrupole focal length f [m]   = ", focal_length)
        println("  bend angle per half cell [rad]  = ", theta)
        println("  horizontal phase advance [deg]  = ", rad2deg(optics.mid_qf.mu))
        println("  beta_x at QF/QD [m]             = ", (optics.mid_qf.beta, optics.mid_qd.beta))
        println("  D_x at QF/QD [m]                = ", (dispersion.mid_qf[1], dispersion.mid_qd[1]))
        println("  natural chromaticity (x, y)     = ", chrom_momentum)
        println("  H at QF/QD [m]                  = ", (h.mid_qf, h.mid_qd))
        println("  note's 100-point optimum [deg]  = ", optimum.grid_degrees)
        println("  continuous optimum [deg]        = ", optimum.degrees)
        println("  all checks passed               = ", check)
    end

    return (;
        beam, lattices, maps, thin, thin_dispersion, analytic, optics,
        dispersion, h, chrom_energy, chrom_momentum, radiation, scan,
        optimum, errors,
    )
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    FODOCellWithDipolesExample.run_example()
end
