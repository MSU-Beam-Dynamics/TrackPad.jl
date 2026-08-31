using Test
using TrackPad
using LinearAlgebra
using Random

@testset "Matched Gaussian distributions" begin
    @testset "Direct 4D and 6D covariance" begin
        lower6 = [
            1.2e-3  0.0     0.0     0.0     0.0     0.0
           -2.0e-4  3.0e-4  0.0     0.0     0.0     0.0
            1.0e-4 -2.0e-5  8.0e-4  0.0     0.0     0.0
            0.0     1.0e-5  1.5e-4  2.5e-4  0.0     0.0
            2.0e-5  0.0     0.0     0.0     3.0e-3  0.0
            0.0     1.0e-5  0.0     0.0    -4.0e-4  7.0e-4
        ]
        sigma6 = lower6 * lower6'
        center6 = [1.0e-3, -2.0e-4, 3.0e-4, 1.0e-4, -5.0e-3, 2.0e-4]
        coordinates6 = gaussian_distribution(
            MersenneTwister(1234), 512, sigma6; centroid=center6,
        )
        measured_center = vec(sum(coordinates6; dims=1)) / size(coordinates6, 1)
        @test size(coordinates6) == (512, 6)
        @test measured_center ≈ center6 atol=2.0e-17
        @test beam_covariance(coordinates6) ≈ sigma6 rtol=2.0e-13 atol=2.0e-20

        sigma4 = sigma6[1:4, 1:4]
        coordinates4 = gaussian_distribution(
            MersenneTwister(4321), 128, sigma4; corrected=true,
        )
        @test size(coordinates4) == (128, 4)
        @test beam_covariance(coordinates4; corrected=true) ≈
              sigma4 rtol=2.0e-13 atol=2.0e-20

        raw = randn(MersenneTwister(99), 256, 6)
        match_moments!(raw, sigma6; centroid=center6)
        @test vec(sum(raw; dims=1)) / size(raw, 1) ≈ center6 atol=2.0e-17
        @test beam_covariance(raw) ≈ sigma6 rtol=2.0e-13 atol=2.0e-20

        uncorrected = gaussian_distribution(
            MersenneTwister(7), 64, sigma4; exact_moments=false,
        )
        @test norm(beam_covariance(uncorrected) - sigma4, Inf) > 1.0e-10
    end

    @testset "Twiss, dispersion, and crab dispersion" begin
        horizontal = optics2D(12.0, -1.2, 0.0, 0.35, -0.04)
        vertical = optics2D(7.0, 0.4, 0.0, -0.08, 0.015)
        optics = optics4DUC(horizontal, vertical)
        emitx = 2.0e-8
        emity = 3.0e-9
        longitudinal = [4.0e-6 -1.0e-7; -1.0e-7 9.0e-8]
        crab = [0.025, -0.003, -0.012, 0.002]
        ordinary = [horizontal.eta, horizontal.etap, vertical.eta, vertical.etap]

        sigma = matched_covariance(
            optics; emitx=emitx, emity=emity,
            longitudinal=longitudinal, crab_dispersion=crab,
        )
        sigma0 = zeros(6, 6)
        sigma0[1:2, 1:2] .= emitx .* [
            horizontal.beta -horizontal.alpha
            -horizontal.alpha horizontal.gamma
        ]
        sigma0[3:4, 3:4] .= emity .* [
            vertical.beta -vertical.alpha
            -vertical.alpha vertical.gamma
        ]
        sigma0[5:6, 5:6] .= longitudinal
        response = Matrix{Float64}(I, 6, 6)
        response[1:4, 5] .= crab
        response[1:4, 6] .= ordinary
        @test sigma ≈ response * sigma0 * response' atol=1.0e-20

        coordinates = matched_gaussian(
            MersenneTwister(2026), 1024, optics;
            emitx=emitx, emity=emity, longitudinal=longitudinal,
            crab_dispersion=crab,
        )
        @test beam_covariance(coordinates) ≈ sigma rtol=3.0e-13 atol=2.0e-20

        keyword_coordinates = matched_gaussian(
            MersenneTwister(5), 128;
            betax=12.0, alphax=-1.2, betay=7.0, alphay=0.4,
            emitx=emitx, emity=emity, emitz=1.0e-7, betaz=0.5,
        )
        @test size(keyword_coordinates) == (128, 6)

        float32_coordinates = matched_gaussian(
            MersenneTwister(6), 64;
            betax=Float32(12), betay=Float32(7),
            emitx=Float32(2e-8), emity=Float32(3e-9),
        )
        @test eltype(float32_coordinates) == Float32

        beam = Beam(1.0e9)
        lost = zeros(Int, size(coordinates, 1))
        linepass!(coordinates, Lattice([Drift(0.1)]), beam, lost)
        @test all(iszero, lost)
    end

    @testset "Projected and eigen-emittances" begin
        optics = optics4DUC(10.0, -0.7, 4.0, 0.3)
        sigma = matched_covariance(
            optics; emitx=4.0e-8, emity=2.0e-9,
            emitz=3.0e-7, betaz=0.2,
            dispersion=zeros(4), crab_dispersion=zeros(4),
        )
        expected = [4.0e-8, 2.0e-9, 3.0e-7]
        @test projected_emittances(sigma) ≈ expected rtol=2.0e-14
        @test eigenemittances(sigma) ≈ sort(expected) rtol=2.0e-13

        angle = 0.37
        c, s = cos(angle), sin(angle)
        coupling = Matrix{Float64}(I, 6, 6)
        coupling[1:2, 1:2] .= c .* Matrix{Float64}(I, 2, 2)
        coupling[1:2, 3:4] .= s .* Matrix{Float64}(I, 2, 2)
        coupling[3:4, 1:2] .= -s .* Matrix{Float64}(I, 2, 2)
        coupling[3:4, 3:4] .= c .* Matrix{Float64}(I, 2, 2)
        coupled_sigma = coupling * sigma * coupling'
        @test eigenemittances(coupled_sigma) ≈ sort(expected) rtol=3.0e-13
        @test projected_emittances(coupled_sigma)[1:2] != expected[1:2]
    end

    @testset "Input validation" begin
        sigma6 = Matrix{Float64}(I, 6, 6)
        @test_throws ArgumentError gaussian_distribution(MersenneTwister(1), 6, sigma6)
        @test_throws ArgumentError gaussian_distribution(MersenneTwister(1), 0, sigma6)
        @test_throws DimensionMismatch gaussian_distribution(
            MersenneTwister(1), 10, Matrix{Float64}(I, 5, 5),
        )
        asymmetric = copy(sigma6)
        asymmetric[1, 2] = 0.1
        @test_throws ArgumentError gaussian_distribution(
            MersenneTwister(1), 10, asymmetric,
        )
        indefinite = copy(sigma6)
        indefinite[1, 1] = -1.0
        @test_throws ArgumentError gaussian_distribution(
            MersenneTwister(1), 10, indefinite,
        )
        @test_throws DimensionMismatch gaussian_distribution(
            MersenneTwister(1), 10, sigma6; centroid=zeros(4),
        )
        rank_deficient = ones(10, 6)
        @test_throws ArgumentError match_moments!(rank_deficient, sigma6)
        @test_throws ArgumentError matched_covariance(
            optics4DUC(-1.0, 0.0, 1.0, 0.0); emitx=1.0, emity=1.0,
        )
        @test_throws DimensionMismatch matched_covariance(
            optics4DUC(1.0, 0.0, 1.0, 0.0);
            emitx=1.0, emity=1.0, dispersion=zeros(3),
        )
    end
end
