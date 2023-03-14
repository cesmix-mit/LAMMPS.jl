# LAMMPS.jl
function compute_fptr end

mutable struct Pair{F, E}
    lmp::LMP
    name::String
    compute_force::F
    compute_energy::E

    function Pair(lmp::LMP, name, compute_force::F, compute_energy::E) where {F, E}
        if haskey(lmp.registered_pairs, name)
            error("Pair has already been registered with $name")
        end

        this = new{F, E}(lmp, name, compute_force, compute_energy)
        lmp.registered_pairs[name] = this # preserves pair globally

        ctx = Base.pointer_from_objref(this)
        callback = @cfunction(compute_fptr, Cvoid, (Ptr{Cvoid}, Cint, Cint))
        API.lammps_foreign_add_pair_style(lmp, name, ctx, callback)

        return this
    end
end

# function lammps_neighlist_num_elements(handle, idx)
#     ccall((:lammps_neighlist_num_elements, liblammps), Cint, (Ptr{Cvoid}, Cint), handle, idx)
# end

# function (handle, idx, element, iatom, numneigh, neighbors)
#     ccall((:lammps_neighlist_element_neighbors, liblammps), Cvoid, (Ptr{Cvoid}, Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Ptr{Cint}}), handle, idx, element, iatom, numneigh, neighbors)
# end

function neighbors(lmp, idx, element)
    r_iatom = Ref{Cint}()
    r_numneigh = Ref{Cint}()
    r_neighbors = Ref{Ptr{Cint}}(0)

    API.lammps_neighlist_element_neighbors(lmp, idx, element - 1, r_iatom, r_numneigh, r_neighbors)

    return r_iatom[], Base.unsafe_wrap(Array, r_neighbors[], r_numneigh[]; own = false)
end

function compute_fptr(ptr::Ptr{Cvoid}, eflag::Cint, vflag::Cint)
    pair = Base.unsafe_pointer_to_objref(ptr)::Pair
    lmp = pair.lmp
    
    # Full neighbor list
    idx = API.lammps_find_pair_neighlist(lmp, pair.name, 1, 0, 0)
    if idx == -1
        error("Could not find neighbor list for pair $(pair.name)")
    end
    
    x = extract_atom(lmp, "x")
    f = extract_atom(lmp, "f")

    nelements = API.lammps_neighlist_num_elements(lmp, idx)
    @debug "Calling compute_fptr on" pair eflag vflag idx nelements

    for ii in 1:nelements
        iatom, neigh = neighbors(lmp, idx, ii)
        iatom += 1
        xtmp, ytmp, ztmp = view(x, :, iatom) # TODO SArray?
        for jj in 1:length(neigh)
            jatom = neigh[jj] + 1
            delx = xtmp - x[1, jatom]
            dely = ytmp - x[2, jatom]
            delz = ztmp = x[3, jatom]
        
            rsq = delx*delx + dely*dely + delz*delz;

            # TODO: rsq < cutsq[itype][jtype]
            fpair = pair.compute_force(rsq, 0, 1) # TODO add function barrier after pair lookup

            f[1, iatom] += delx*fpair
            f[2, iatom] += dely*fpair
            f[3, iatom] += delz*fpair
            # if (newton_pair || j < nlocal) {
            #   f[j][0] -= delx*fpair;
            #   f[j][1] -= dely*fpair;
            #   f[j][2] -= delz*fpair;
            # }
            # TODO eflag 
            # TODO evflag
        end
    end

    return nothing
end




# function compute_energy(pair::Pair#=...=#)
#     for i in 1:10
#         itype ...
#         for j in 1:10
#             jtype = ...
#             rsq = ...
#             pair.compute_energy(rsq, itype, jtype)
#             # update values
#         end
#     end
# end

# # User code:
# # What user writes
# function my_compute_energy(coeff, rsq, itype, jtype)
#     coeff = self.coeff[self.pmap[itype]][self.pmap[jtype]]
#     r2inv  = 1.0/rsq
#     r6inv  = r2inv*r2inv*r2inv
#     lj3 = coeff[2]
#     lj4 = coeff[3]
#     return (r6inv * (lj3*r6inv - lj4))
# end

# const coeff = []
# myPair = Pair(my_compute_force, (rsq, itype, jtype)->my_compute_energy(coeff, rsq, itype, jtype))
# lmp = ...
# LAMMPS.add_foreign_pair!("myPair", myPair)

# # ... command
# command(lmp, "pair_style myPair")
# # ... command