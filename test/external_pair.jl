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


@testset "external_pair" begin
    for units in ("lj", "si"),  backend in (nothing, AutoForwardDiff(), AutoEnzyme())
        lmp_native = LMP(["-screen", "none"])
        lmp_julia = LMP(["-screen", "none"])

        for lmp in (lmp_native, lmp_julia)
            command(lmp, """
                units $units
                atom_style atomic
                box tilt large
                atom_modify map array sort 0 0

                boundary p p p
                region cell block 0 10.0 0 10.0 0 10.0 units box
                create_box 1 cell
            """)
        end

        cutoff = 2.5
        command(lmp_native, "pair_style lj/cut $cutoff")
        command(lmp_native, "pair_coeff * * 1 1")

        # Register external fix
        lj = LAMMPS.PairExternal(lmp_julia, "julia_lj", cutoff; backend) do r, itype, jtype
            ε, σ = coefficients[itype][jtype]
            r6inv  = (σ/r)^6
            energy = 4ε * (r6inv * (r6inv - 1))
            force = 24ε * (r6inv * (2r6inv - 1)) / r
            return backend === nothing ? (energy, force) : energy
        end

        # Setup atoms
        natoms = 10
        positions = rand(3, 10) .* 5
        for lmp in (lmp_native, lmp_julia)
            command(lmp, """
                create_atoms 1 random $natoms 1 NULL
                mass 1 1.0
                compute potential all pe/atom
                compute virial all stress/atom NULL
                compute virial_tot all pressure thermo_temp
            """)

            scatter!(lmp, "x", positions)

            command(lmp, "run 0")
        end

        # extract forces
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

        @testset "$units $backend" begin
            @test potential_native ≈ potential_julia
            @test forces_native ≈ forces_julia
            @test virial_native ≈ virial_julia
            @test virial_tot_native ≈ virial_tot_julia
            @test energy_native ≈ energy_julia
        end
    end
end