using LAMMPS
using Test
using ADTypes
import ForwardDiff
import Enzyme

const coefficients = Base.ImmutableDict(
    1 => Base.ImmutableDict(
        1 => (1.0, 1.0)
    )
)

positions = [
    3.65715  3.38985  2.563    9.07901   1.88995  7.67364  8.09311  8.12831  4.78828  3.72213  4.50852  0.327834  1.0418   6.39688   9.70606  7.69569  3.40795   5.29387   9.66779  6.12549
    2.28259  5.95683  4.72931  3.32801   2.36841  1.86678  1.35607  3.40432  5.43864  2.05714  7.57347  6.63626   2.32314  0.589051  1.41243  1.98314  0.691965  0.383908  7.05885  8.1152
    2.04191  4.97413  6.53093  0.990347  2.00388  2.41298  5.5144   5.98437  3.61306  4.09874  7.20735  3.07125   4.099    7.71419   5.83603  5.44341  5.48853   4.05366   1.81701  6.75289
]

function test_interaction(lmp_native, lmp_julia, testset)
    for lmp in (lmp_native, lmp_julia)
        command(lmp, """
            compute potential all pe/atom
            compute virial all stress/atom NULL
            compute virial_tot all pressure thermo_temp
            run 0
        """)
    end

    potential_native = gather(lmp_native, "c_potential", Float64)
    potential_julia = gather(lmp_julia, "c_potential", Float64)
    forces_native = gather(lmp_native, "f", Float64)
    forces_julia = gather(lmp_julia, "f", Float64)
    virial_native = gather(lmp_native, "c_virial", Float64)
    virial_julia = gather(lmp_julia, "c_virial", Float64)
    virial_tot_native = extract_compute(lmp_native, "virial_tot", STYLE_GLOBAL, TYPE_VECTOR)
    virial_tot_julia = extract_compute(lmp_julia, "virial_tot", STYLE_GLOBAL, TYPE_VECTOR)
    energy_native = LAMMPS.API.lammps_get_thermo(lmp_native, "etotal")
    energy_julia = LAMMPS.API.lammps_get_thermo(lmp_julia, "etotal")

    @testset "$testset" begin
        @test potential_native ≈ potential_julia
        @test forces_native ≈ forces_julia
        @test virial_native ≈ virial_julia
        @test virial_tot_native ≈ virial_tot_julia
        @test energy_native ≈ energy_julia
    end
end

@testset "external_pair_lj" begin
    for units in ("lj", "si"),  backend in (nothing, AutoForwardDiff(), AutoEnzyme())
        lmp_native = LMP(["-screen", "none"])
        lmp_julia = LMP(["-screen", "none"])

        for lmp in (lmp_native, lmp_julia)
            command(lmp, """
                units $units
                atom_style atomic
                atom_modify map array sort 0 0

                boundary p p p
                region cell block 0 10 0 10 0 10 units box
                create_box 1 cell
            """)
        end

        cutoff = 2.5
        command(lmp_native, "pair_style lj/cut $cutoff")
        command(lmp_native, "pair_coeff * * 1 1")

        # Register external fix
        config = InteractionConfig(
            atom = @NamedTuple{type::Int32},
            backend = backend,
        )

        PairExternal(lmp_julia, config, cutoff) do r, system, iatom, jatom
            ε, σ = coefficients[iatom.type][jatom.type]
            r6inv  = (σ/r)^6
            energy = 4ε * (r6inv * (r6inv - 1))
            force = 24ε * (r6inv * (2r6inv - 1)) / r
            return backend === nothing ? (energy, force) : energy
        end

        for lmp in (lmp_native, lmp_julia)
            command(lmp, """
                create_atoms 1 random 20 1 NULL
                mass 1 1.0
            """)

            scatter!(lmp, "x", positions)
        end

        test_interaction(lmp_native, lmp_julia, "$units $backend")
    end
end

@testset "external_pair_coul" begin
    for units in ("lj", "si"),  backend in (nothing, AutoForwardDiff(), AutoEnzyme())
        lmp_native = LMP(["-screen", "none"])
        lmp_julia = LMP(["-screen", "none"])

        for lmp in (lmp_native, lmp_julia)
            command(lmp, """
                units $units
                atom_style charge
                atom_modify map array sort 0 0

                boundary p p p
                region cell block 0 10 0 10 0 10 units box
                create_box 1 cell
            """)
        end

        cutoff = 2.5
        command(lmp_native, "pair_style coul/cut $cutoff")
        command(lmp_native, "pair_coeff * *")

        config = InteractionConfig(
            system = @NamedTuple{qqrd2e::Float64},
            atom = @NamedTuple{q::Float64},
            backend = backend,
        )

        PairExternal(lmp_julia, config, cutoff) do r, system, iatom, jatom
            energy = system.qqrd2e * (iatom.q * jatom.q) / r
            force = system.qqrd2e * (iatom.q * jatom.q) / r^2
            return backend === nothing ? (energy, force) : energy
        end

        for lmp in (lmp_native, lmp_julia)
            command(lmp, """
                create_atoms 1 random 20 1 NULL
                set type 1 charge 2.0
                mass 1 1.0
            """)

            scatter!(lmp, "x", positions)
        end

        test_interaction(lmp_native, lmp_julia, "$units $backend")
    end
end

@testset "external_bond_harmonic" begin
    file = joinpath(@__DIR__, "test_files/bonds_angles_dihedrals_impropers.data")

    for units in ("lj", "si"),  backend in (nothing, AutoForwardDiff(), AutoEnzyme())
        lmp_native = LMP(["-screen", "none"])
        lmp_julia = LMP(["-screen", "none"])

        for lmp in (lmp_native, lmp_julia)
            command(lmp, """
                units $units
                atom_style molecular
                newton off
                read_data $file
                pair_style none
            """)
        end

        config = InteractionConfig(
            system = @NamedTuple{},
            atom = @NamedTuple{},
            backend = backend,
        )

        BondExternal(lmp_julia, config) do r, type, system, iatom, jatom
            energy = (r-1)^2
            force = -2(r-1)
            return backend === nothing ? (energy, force) : energy
        end

        command(lmp_native, """
            bond_style harmonic
            bond_coeff * 1 1
        """)

        test_interaction(lmp_native, lmp_julia, "$units $backend")
    end
end

@testset "external_angle_cosine" begin
    file = joinpath(@__DIR__, "test_files/bonds_angles_dihedrals_impropers.data")

    for units in ("lj", "si"),  backend in (AutoForwardDiff(), AutoEnzyme())
        lmp_native = LMP(["-screen", "none"])
        lmp_julia = LMP(["-screen", "none"])

        for lmp in (lmp_native, lmp_julia)
            command(lmp, """
                units $units
                atom_style molecular
                newton off
                read_data $file
                pair_style none
            """)
        end

        config = InteractionConfig(
            system = @NamedTuple{},
            atom = @NamedTuple{},
            backend = backend,
        )

        AngleExternal(lmp_julia, config) do θ, type, system, iatom, jatom, katom
            return 1+cos(θ)
        end

        command(lmp_native, """
            angle_style cosine
            angle_coeff * 1
        """)

        test_interaction(lmp_native, lmp_julia, "$units $backend")
    end
end

@testset "external_angle_harmonic" begin
    file = joinpath(@__DIR__, "test_files/bonds_angles_dihedrals_impropers.data")

    for units in ("lj", "si"),  backend in (AutoForwardDiff(), AutoEnzyme())
        lmp_native = LMP(["-screen", "none"])
        lmp_julia = LMP(["-screen", "none"])

        for lmp in (lmp_native, lmp_julia)
            command(lmp, """
                units $units
                atom_style molecular
                newton off
                read_data $file
                pair_style none
            """)
        end

        config = InteractionConfig(
            system = @NamedTuple{},
            atom = @NamedTuple{},
            backend = backend,
        )

        AngleExternal(lmp_julia, config) do θ, type, system, iatom, jatom, katom
            return (θ-deg2rad(45))^2
        end

        command(lmp_native, """
            angle_style harmonic
            angle_coeff * 1 45
        """)

        test_interaction(lmp_native, lmp_julia, "$units $backend")
    end
end