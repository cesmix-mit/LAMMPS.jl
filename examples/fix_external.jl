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
lj = LAMMPS.PairExternal(lmp, "julia_lj", "zero", compute_force, compute_energy, cutoff)

# Setup atoms
natoms = 10
command(lmp, "create_atoms 1 random $natoms 1 NULL")
command(lmp, "mass 1 1.0")

# (x,y,z), natoms
# positions = rand(3, 10) .* 5
positions = [4.4955289268519625 3.3999909266656836 4.420245465344918 2.3923580632470216 1.9933183377321746 2.3367019702697096 0.014668174434679937 4.5978923623562356 2.9389893820585025 4.800351333939365; 4.523573662784505 3.1582899538900304 2.5562765646443 3.199496583966941 4.891026316235915 4.689641854106464 2.7591724192198575 0.7491156338926308 1.258994308308421 2.0419941687773937; 2.256261603545908 0.694847945108647 4.058244561946366 3.044596885569421 2.60225212714946 4.0030490608195555 0.9941423774290642 1.8076536961230087 1.9712395260164222 1.2705916409499818]

LAMMPS.API.lammps_scatter_atoms(lmp, "x", 1, 3, positions)

command(lmp, "run 0")

# extract forces
forces = extract_atom(lmp, "f")
