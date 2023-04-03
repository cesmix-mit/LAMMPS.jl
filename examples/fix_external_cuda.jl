using LAMMPS
using CUDA
using Adapt
using ArraysOfArrays

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
command(lmp, "fix julia_lj_cuda all external pf/callback 1 1")

struct Potential{Coeff}
    coefficients::Coeff
end

function compute_force(p::Potential, rsq, itype, jtype)
    coeff = p.coefficients[itype][jtype]
    r2inv  = 1.0/rsq
    r6inv  = r2inv^3
    lj1 = coeff[1]
    lj2 = coeff[2]
    return (r6inv * (lj1*r6inv - lj2))*r2inv
end

function compute_energy(p::Potential, rsq, itype, jtype)
    coeff = p.coefficients[itype][jtype]
    r2inv  = 1.0/rsq
    r6inv  = r2inv^3
    lj3 = coeff[3]
    lj4 = coeff[4]
    return (r6inv * (lj3*r6inv - lj4))
end

const NEIGHMASK = 0x3FFFFFFF
const SBBITS = 30
sbmask(atom) = (atom >> SBBITS) & 3
const special_lj = CuArray([1.0, 0.0, 0.0 ,0.0])

LAMMPS.FixExternal(lmp, "julia_lj_cuda") do fix, timestep, nlocal, ids, x, fexternal
    # Full neighbor list
    idx = LAMMPS.pair_neighbor_list(fix.lmp, "zero", 1, 0, 0)
    nelements = LAMMPS.API.lammps_neighlist_num_elements(fix.lmp, idx)

    potential = Potential((((48.0, 24.0, 4.0, 4.0),),))
    cutsq::Float64 = cutoff^2

    # TODO how to obtain in fix
    eflag = false
    evflag = false

    newton_pair = LAMMPS.extract_setting(fix.lmp, "newton_pair") == 1
    # special_lj = extract_global(fix.lmp, "special_lj")

    type = LAMMPS.extract_atom(lmp, "type")

    # zero-out fexternal (noticed some undef memory)
    fexternal .= 0

    energies = CUDA.zeros(nlocal)
    cu_fexternal = adapt(CuArray, fexternal)
    cu_x = adapt(CuArray, x)
    cu_type = adapt(CuArray, type)

    neighbors_array = VectorOfArrays{Int64, 2}()
    ilist = Vector{Int64}()

    # Copy neighbor_list to Julia datastructure
    for ii in 1:nelements
        # local atom index (i.e. in the range [0, nlocal + nghost)
        iatom, neigh = LAMMPS.neighbors(lmp, idx, ii)
        push!(neighbors_array, reshape(neigh, (1, length(neigh))))
        push!(ilist, iatom)
    end
    neighbors_array = adapt(CuArray, neighbors_array)
    ilist = adapt(CuArray, ilist)

    function kernel(potential, x, fexternal, energies, ilist, neighbors, cutsq, nlocal, type, special_lj)
        ii = threadIdx().x
        iatom = ilist[ii]
        neighs = neighbors[ii]

        iatom += 1 # 1-based indexing
        xtmp = x[1, iatom]
        ytmp = x[2, iatom]
        ztmp = x[3, iatom]

        itype = type[iatom]

        for jatom in neighs
            factor_lj = special_lj[sbmask(jatom) + 1]
            jatom &= NEIGHMASK
            jatom += 1 # 1-based indexing

            delx = xtmp - x[1, jatom]
            dely = ytmp - x[2, jatom]
            delz = ztmp - x[3, jatom]

            jtype = type[jatom]

            rsq = delx*delx + dely*dely + delz*delz;
            if rsq < cutsq
                fpair = factor_lj * compute_force(potential, rsq, itype, jtype)

                if iatom <= nlocal
                    CUDA.@atomic fexternal[1, iatom] += delx*fpair
                    CUDA.@atomic fexternal[2, iatom] += dely*fpair
                    CUDA.@atomic fexternal[3, iatom] += delz*fpair
                    if jatom <= nlocal || newton_pair
                        CUDA.@atomic fexternal[1, jatom] -= delx*fpair
                        CUDA.@atomic fexternal[2, jatom] -= dely*fpair
                        CUDA.@atomic fexternal[3, jatom] -= delz*fpair
                    end
                    energies[iatom] += compute_energy(potential, rsq, itype, jtype)
                end
            end
        end
        return nothing
    end

    @cuda threads=nlocal kernel(potential, cu_x, cu_fexternal, energies,
                                ilist, neighbors_array,
                                cutsq, nlocal, cu_type, special_lj)

    copyto!(fexternal, cu_fexternal) # TODO async
    total_energy = sum(energies)
    energies = Array(energies)

    LAMMPS.API.lammps_fix_external_set_energy_peratom(fix.lmp, fix.name, energies)
    LAMMPS.energy_global!(fix, total_energy)
end

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
