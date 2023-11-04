using LAMMPS
using Test

lmp_native = LMP()
lmp_julia = LMP()

for lmp in (lmp_native, lmp_julia)
    command(lmp, "units lj")
    command(lmp, "atom_style atomic")
    command(lmp, "atom_modify map array sort 0 0")
    command(lmp, "box tilt large")

    # Setup box
    command(lmp, "boundary p p p")
    command(lmp, "region cell block 0 10.0 0 10.0 0 10.0 units box")
    command(lmp, "create_box 1 cell")
end

cutoff = 2.5
command(lmp_julia, "pair_style zero $cutoff")
command(lmp_julia, "pair_coeff * *")
command(lmp_julia, "fix julia_lj all external pf/callback 1 1")

command(lmp_native, "pair_style lj/cut $cutoff")
command(lmp_native, "pair_coeff * * 1 1")

const coefficients = Dict(
    1 => Dict(
        1 => [48.0, 24.0, 4.0,4.0]
    )
)

function compute_force(rsq, itype, jtype)
    coeff = coefficients[itype][jtype]
    r2inv  = 1.0/rsq
    r6inv  = r2inv^3
    lj1 = coeff[1]
    lj2 = coeff[2]
    return (r6inv * (lj1*r6inv - lj2))*r2inv
end

function compute_energy(rsq, itype, jtype)
    coeff = coefficients[itype][jtype]
    r2inv  = 1.0/rsq
    r6inv  = r2inv^3
    lj3 = coeff[3]
    lj4 = coeff[4]
    return (r6inv * (lj3*r6inv - lj4))
end

# Register external fix
lj = LAMMPS.PairExternal(lmp_julia, "julia_lj", "zero", compute_force, compute_energy, cutoff)

# Setup atoms
natoms = 10
positions = rand(3, 10) .* 5
for lmp in (lmp_native, lmp_julia)
    command(lmp, "create_atoms 1 random $natoms 1 NULL")
    command(lmp, "mass 1 1.0")

    LAMMPS.API.lammps_scatter_atoms(lmp, "x", 1, 3, positions)

    command(lmp, "run 0")
end

# extract forces
forces_native = extract_atom(lmp_native, "f")
forces_julia = extract_atom(lmp_julia, "f")

@testset "External Pair" begin
    @test forces_native == forces_julia
end
