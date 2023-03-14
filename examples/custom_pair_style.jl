using LAMMPS

lmp = LMP()

function compute_force(rsq, itype, jtype)
    @show (rsq, itype, jtype)
    return one(rsq)
end

# Register pair
LAMMPS.Pair(lmp, "julia_lj", compute_force, identity)

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

command(lmp, "pair_style julia_lj 2.5")
command(lmp, "pair_coeff * * 1 1") # TODO

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