function fix_external_callback end

mutable struct FixExternal
    lmp::LMP
    name::String
    callback

    function FixExternal(lmp::LMP, name::String, callback)
        if haskey(lmp.external_fixes, name)
            error("FixExternal has already been registered with $name")
        end

        this = new(lmp, name, callback)
        lmp.external_fixes[name] = this # preserves pair globally

        ctx = Base.pointer_from_objref(this)
        callback = @cfunction(fix_external_callback, Cvoid, (Ptr{Cvoid}, Int64, Cint, Ptr{Cint}, Ptr{Ptr{Float64}}, Ptr{Ptr{Float64}}))
        API.lammps_set_fix_external_callback(lmp, name, callback, ctx)

        return this
    end
end

FixExternal(callback, lmp::LMP, name::String) = FixExternal(lmp::LMP, name::String, callback)

function fix_external_callback(ctx::Ptr{Cvoid}, timestep::Int64, nlocal::Cint, ids::Ptr{Cint}, x::Ptr{Ptr{Float64}}, fexternal::Ptr{Ptr{Float64}})
    fix = Base.unsafe_pointer_to_objref(ctx)::FixExternal
    nlocal = Int(nlocal)

    @debug "Calling fix_external_callback on" fix timestep nlocal
    shape = (nlocal, 3)
    x = unsafe_wrap(x, shape)
    fexternal = unsafe_wrap(fexternal, shape)
    ids = unsafe_wrap(ids, (nlocal,))

    fix.callback(fix, timestep, nlocal, ids, x, fexternal)
    return nothing
end

function energy_global!(fix::FixExternal, energy)
    API.lammps_fix_external_set_energy_global(fix.lmp, fix.name, energy)
end

function neighbor_list(fix::FixExternal, request)
    fix_neighbor_list(fix.lmp, fix.name, request)
end

# TODO
# virial_global!
# function virial_global!(fix::FixExternal, )

const NEIGHMASK = 0x3FFFFFFF
const SBBITS = 30
sbmask(atom) = (atom >> SBBITS) & 3
const special_lj = [1.0, 0.0, 0.0 ,0.0]

function PairExternal(lmp, name, neigh_name, compute_force, compute_energy, cut_global)
    cutsq = cut_global^2
    FixExternal(lmp, name) do fix, timestep, nlocal, ids, x, fexternal
        # Full neighbor list
        idx = pair_neighbor_list(fix.lmp, neigh_name, 1, 0, 0)
        nelements = API.lammps_neighlist_num_elements(fix.lmp, idx)

        # TODO how to obtain in fix
        eflag = false
        evflag = false

        newton_pair = extract_setting(fix.lmp, "newton_pair") == 1
        # special_lj = extract_global(fix.lmp, "special_lj")

        type = LAMMPS.extract_atom(lmp, "type")

        # zero-out fexternal (noticed some undef memory)
        fexternal .= 0

        energies = zeros(nlocal)

        for ii in 1:nelements
            # local atom index (i.e. in the range [0, nlocal + nghost)
            iatom, neigh = LAMMPS.neighbors(lmp, idx, ii)
            iatom += 1 # 1-based indexing
            xtmp, ytmp, ztmp = view(x, :, iatom) # TODO SArray?
            itype = type[iatom]
            for jj in 1:length(neigh)
                jatom = neigh[jj]
                factor_lj = special_lj[sbmask(jatom) + 1]
                jatom &= NEIGHMASK
                jatom += 1 # 1-based indexing

                delx = xtmp - x[1, jatom]
                dely = ytmp - x[2, jatom]
                delz = ztmp - x[3, jatom]
                jtype = type[jatom]

                rsq = delx*delx + dely*dely + delz*delz;
                if rsq < cutsq
                    fpair = factor_lj * compute_force(rsq, itype, jtype)

                    if iatom <= nlocal
                        fexternal[1, iatom] += delx*fpair
                        fexternal[2, iatom] += dely*fpair
                        fexternal[3, iatom] += delz*fpair
                        if jatom <= nlocal || newton_pair
                            fexternal[1, jatom] -= delx*fpair
                            fexternal[2, jatom] -= dely*fpair
                            fexternal[3, jatom] -= delz*fpair
                        end
                        energies[iatom] += compute_energy(rsq, itype, jtype)
                    end
                end
            end
        end
        API.lammps_fix_external_set_energy_peratom(fix.lmp, fix.name, energies)
        energy_global!(fix, sum(energies))
    end
end
