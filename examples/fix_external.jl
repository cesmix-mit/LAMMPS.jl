using LAMMPS
using Test
using AtomsCalculators: potential_energy, forces
using AtomsBase
using AtomsBuilder
using ACEpotentials
using ExtXYZ
using AtomsBase: AbstractSystem
using LinearAlgebra: norm
using Unitful

lmp = LMP(["-screen", "none"])


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

if !isfile("Si_dataset.xyz")
    download("https://www.dropbox.com/scl/fi/z6lvcpx3djp775zenz032/Si-PRX-2018.xyz?rlkey=ja5e9z99c3ta1ugra5ayq5lcv&st=cs6g7vbu&dl=1",
         "Si_dataset.xyz");
end

Si_dataset = ExtXYZ.load("Si_dataset.xyz");

Si_tiny_dataset, _, _ = ACEpotentials.example_dataset("Si_tiny");

deleteat!(Si_dataset, 1);

hyperparams = (elements = [:Si,],
               order = 3,
               totaldegree = 8,
               rcut = 2.5,
               Eref = Dict(:Si => -158.54496821))
model = ACEpotentials.ace1_model(;hyperparams...);
solver = ACEfit.QR(lambda=1e-1)
data_keys = (energy_key = "dft_energy", force_key = "dft_force", virial_key = "dft_virial")
acefit!(Si_tiny_dataset, model;
        solver=solver, data_keys...);
labelmap = Dict(1 => :Si)

@inline function compute_force(pos, types)
    sys_size = size(pos, 2)
    particles = [AtomsBase.Atom(ChemicalSpecies(labelmap[types[i]]), pos[:, i].*u"Å") for i in 1:sys_size]
    cell = AtomsBuilder.bulk(:Si, cubic=true) * 3
    sys = FlexibleSystem(particles, cell)
    f = forces(sys, model)
    return ustrip.(f)
end

@inline function compute_energy(pos, types)
    sys_size = size(pos, 2)
    particles = [AtomsBase.Atom(ChemicalSpecies(labelmap[types[i]]), pos[:, i].*u"Å") for i in 1:sys_size]
    cell = AtomsBuilder.bulk(:Si, cubic=true) * 3
    sys = FlexibleSystem(particles, cell)
    energy = potential_energy(sys, model)  
    return ustrip(energy)
end

# Register external fix
lj = LAMMPS.PairExternal(lmp, "julia_lj", "zero", compute_force, compute_energy, cutoff, true, true)

# Setup atoms
natoms = 54
command(lmp, "labelmap atom 1 Si")
command(lmp, "create_atoms Si random $natoms 1 NULL")
positions = rand(3, natoms) .* 5
command(lmp, "mass 1 28.0855")
LAMMPS.scatter!(lmp, "x", positions)

command(lmp, "run 0")

# extract forces
forces_julia = gather(lmp, "f", Float64)

particles = [AtomsBase.Atom(ChemicalSpecies(labelmap[1]), positions[:, i].*u"Å") for i in 1:54]
cell = AtomsBuilder.bulk(:Si, cubic=true) * 3
sys = FlexibleSystem(particles, cell)
f = forces(sys, model)

