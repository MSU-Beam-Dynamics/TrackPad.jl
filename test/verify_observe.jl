using Test
using TrackPad
using StaticArrays

# `track!` watching the bunch as beam position monitors do: after chosen
# elements, on every turn, the particles still alive and the sums of their x
# and y, serial or threaded.

const beam = Beam(3.0e9)
const cell = AbstractElement[
    Quadrupole(0.5, 0.9; num_int_steps=4), Drift(0.6),
    SBend(1.2, 2π/40; num_int_steps=4), Drift(0.6),
    Sextupole(0.2, 2.0; num_int_steps=2),
    Quadrupole(0.5, -0.9; num_int_steps=4), Drift(0.6),
    SBend(1.2, 2π/40; num_int_steps=4), Drift(0.6),
]
const ring = Lattice(vcat([copy(cell) for _ in 1:20]...); periodic=true)

function bunch(n)
    coords = zeros(n, 6)
    for i in 1:n
        coords[i, 1] = 1.0e-4 * sin(i)
        coords[i, 3] = 0.5e-4 * cos(i)
        coords[i, 6] = 1.0e-5 * (i - n / 2) / n
    end
    return coords
end

# What the observation should be: the ring cut after the observed elements,
# each stretch tracked by `track!` in turn, and the survivors summed by hand.
function by_hand(lat, coords, nturns, observe)
    out = zeros(3, length(observe), nturns)
    bounds = [0; observe; length(lat.elements)]
    stretches = [Lattice(lat.elements[bounds[k] + 1:bounds[k + 1]]) for k in 1:length(bounds) - 1]
    lost = fill(false, size(coords, 1))
    for n in 1:nturns, (k, stretch) in enumerate(stretches)
        isempty(stretch.elements) || track!(coords, stretch, beam; lost=lost)
        k > length(observe) && continue
        alive = findall(.!lost)
        sx, sy = 0.0, 0.0
        for i in alive            # in order, as track! adds them
            sx += coords[i, 1]
            sy += coords[i, 3]
        end
        out[:, k, n] .= (length(alive), sx, sy)
    end
    return out
end

@testset "track! observes the bunch where it is told to" begin
    n, nturns = 96, 6
    observe = [1, 9, 40, 90, length(ring.elements)]
    coords = bunch(n)
    coords[5, 1] = 2.0            # lost at once, beyond the 1 m limit
    coords[17, 6] = 0.4           # lost a while later
    serial = copy(coords)
    observed = zeros(3, length(observe), nturns)
    track!(serial, ring, beam; nturns=nturns, observe=observe, observed=observed)
    hand = copy(coords)
    want = by_hand(ring, hand, nturns, observe)
    @test observed == want
    @test isequal(serial, hand)
    @test observed[1, 1, 1] == n - 1 && observed[1, end, end] < n - 1   # the losses counted

    # Threaded: the same particles, the sums to rounding (each thread's chunk first).
    threaded = copy(coords)
    seen = fill(NaN, 3, length(observe), nturns)
    track!(threaded, ring, beam; nturns=nturns, observe=observe, observed=seen, threaded=true)
    @test isequal(threaded, serial)
    @test seen[1, :, :] == observed[1, :, :]
    @test seen[2:3, :, :] ≈ observed[2:3, :, :] rtol=1e-12

    # Without `observe`, nothing changes.
    plain = copy(coords)
    track!(plain, ring, beam; nturns=nturns)
    @test isequal(plain, serial)
end

@testset "observation with radiation and quantum excitation" begin
    radiating = Lattice(AbstractElement[
        e isa SBend ? SBend(e.L, e.angle; num_int_steps=4, rad=2) : e for e in ring.elements];
        periodic=true)
    observe = collect(10:10:length(radiating.elements))
    coords = bunch(64)
    serial, threaded = copy(coords), copy(coords)
    a = zeros(3, length(observe), 4)
    b = zeros(3, length(observe), 4)
    track!(serial, radiating, beam; nturns=4, radiation_seed=7, observe=observe, observed=a)
    track!(threaded, radiating, beam; nturns=4, radiation_seed=7, observe=observe, observed=b,
           threaded=true)
    @test isequal(serial, threaded)      # the photons are each particle's, whichever thread
    @test a[1, :, :] == b[1, :, :]
    @test a[2:3, :, :] ≈ b[2:3, :, :] rtol=1e-12
end

@testset "observation arguments" begin
    coords = bunch(8)
    @test_throws ArgumentError track!(copy(coords), ring, beam; observe=[3])
    @test_throws ArgumentError track!(copy(coords), ring, beam; observe=[3], observed=zeros(3, 1, 2))
    @test_throws ArgumentError track!(copy(coords), ring, beam; observe=[5, 3], observed=zeros(3, 2, 1))
    @test_throws ArgumentError track!(copy(coords), ring, beam; observe=[0], observed=zeros(3, 1, 1))
    @test_throws ArgumentError track!(copy(coords), ring, beam;
                                      observe=[length(ring.elements) + 1], observed=zeros(3, 1, 1))
end
