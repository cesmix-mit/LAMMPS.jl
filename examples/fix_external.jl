using LAMMPS

lmp = LMP()

command(lmp, "units lj")
command(lmp, "atom_style atomic")
command(lmp, "atom_modify map array sort 0 0")
command(lmp, "box tilt large")

# Setup box
x_hi = 10.0
y_hi = 10.0
z_hi = 10.0
command(lmp, "boundary p p p")
command(lmp, "region cell block 0 $x_hi 0 $y_hi 0 $z_hi units box")
command(lmp, "create_box 1 cell")

# Use `pair_style zero` to create neighbor list for `julia_lj`
cutoff = 2.5
command(lmp, "pair_style zero $cutoff")
command(lmp, "pair_coeff * *")
command(lmp, "fix julia_lj all external pf/callback 1 1")

const coefficients = Base.ImmutableDict(
    1 => Base.ImmutableDict(
        1 => [48.0, 24.0, 4.0, 4.0]
    )
)

@inline function compute_force(rsq, itype, jtype)
    coeff = coefficients[itype][jtype]
    r2inv  = 1.0/rsq
    r6inv  = r2inv^3
    lj1 = coeff[1]
    lj2 = coeff[2]
    return (r6inv * (lj1*r6inv - lj2))*r2inv
end

@inline function compute_energy(rsq, itype, jtype)
    coeff = coefficients[itype][jtype]
    r2inv  = 1.0/rsq
    r6inv  = r2inv^3
    lj3 = coeff[3]
    lj4 = coeff[4]
    return (r6inv * (lj3*r6inv - lj4))
end

# Register external fix
lj = LAMMPS.PairExternal(lmp, "julia_lj", "zero", compute_force, compute_energy, cutoff)

# Setup atoms
natoms = 10
command(lmp, "create_atoms 1 random $natoms 1 NULL")
command(lmp, "mass 1 1.0")

# (x,y,z), natoms
positions = rand(3, 10) .* 5

LAMMPS.API.lammps_scatter_atoms(lmp, "x", 1, 3, positions)

command(lmp, "run 0")

# extract forces
forces = extract_atom(lmp, "f")
