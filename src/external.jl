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

        # Ensure function is compiled before timestep 0
        if !precompile(this.callback, (FixExternal, Int, Int, Int, Vector{Int32}, Matrix{Float64}, Matrix{Float64}))
            @warn "Failed to precompile the callback" this.callback
        end
        return this
    end
end

FixExternal(callback, lmp::LMP, name::String) = FixExternal(lmp::LMP, name::String, callback)

function fix_external_callback(ctx::Ptr{Cvoid}, timestep::Int64, nlocal::Cint, ids::Ptr{Cint}, x::Ptr{Ptr{Float64}}, fexternal::Ptr{Ptr{Float64}})
    fix = Base.unsafe_pointer_to_objref(ctx)::FixExternal
    nlocal = Int(nlocal)

    nghost = Int(extract_global(fix.lmp, "nghost"))

    @debug "Calling fix_external_callback on" fix timestep nlocal
    shape = (nlocal+nghost, 3)
    x = unsafe_wrap(x, shape)
    fexternal = unsafe_wrap(fexternal, shape)
    ids = unsafe_wrap(ids, (nlocal+nghost,))

    # necessary dynamic
    fix.callback(fix, timestep, nlocal, nghost, ids, x, fexternal)
    return nothing
end

function energy_global!(fix::FixExternal, energy)
    API.lammps_fix_external_set_energy_global(fix.lmp, fix.name, energy)
end

function neighbor_list(fix::FixExternal, request)
    fix_neighbor_list(fix.lmp, fix.name, request)
end

function virial_global!(fix::FixExternal, virial)
    API.lammps_fix_external_set_virial_global(fix.lmp, fix.name, virial)
end

# TODO
# virial_global!
# function virial_global!(fix::FixExternal, )

const NEIGHMASK = 0x3FFFFFFF
const SBBITS = 30
sbmask(atom) = (atom >> SBBITS) & 3
const special_lj = [1.0, 0.0, 0.0 ,0.0]

function virial_fdotr_compute(fexternal::Matrix{Float64}, x::Matrix{Float64}, nall)
    #TODO: discuss include_group flag
    virial = Array{Float64}(undef, 6)
    for i in 1:nall
        virial[1] = fexternal[1, i] * x[1, i]
        virial[2] = fexternal[2, i] * x[2, i]
        virial[3] = fexternal[3, i] * x[3, i]
        virial[1] = fexternal[2, i] * x[1, i]
        virial[2] = fexternal[3, i] * x[1, i]
        virial[3] = fexternal[3, i] * x[2, i]
    end
    return virial
end

function PairExternal(lmp, name, neigh_name, compute_force::F, compute_energy::E, cut_global, eflag, vflag) where {E, F}
    cutsq = cut_global^2
    function pair(fix::FixExternal, timestep::Int, nlocal::Int, nghost::Int, ids::Vector{Int32}, x::Matrix{Float64}, fexternal::Matrix{Float64})
        # Full neighbor list

        idx = pair_neighbor_list(fix.lmp, neigh_name, 1, 0, 0)
        nelements = API.lammps_neighlist_num_elements(fix.lmp, idx)
        newton_pair = extract_setting(fix.lmp, "newton_pair") == 1
        # special_lj = extract_global(fix.lmp, "special_lj")
        type = LAMMPS.extract_atom(lmp, "type", API.LAMMPS_INT, nlocal+nghost)::Vector{Int32}

        # zero-out fexternal (noticed some undef memory)
        fexternal .= 0
        
        energies = zeros(nlocal)


        #API.lammps_fix_external_set_energy_peratom(fix.lmp, fix.name, energies)
        x = gather(lmp, "x", Float64)

        for ii in 1:Int(nelements)
            # local atom index (i.e. in the range [0, nlocal + nghost)
            types = []
            iatom, neigh = LAMMPS.neighbors(lmp, idx, ii) 
            pt = []
            iatom += 1 # 1-based indexing
            xtmp, ytmp, ztmp = view(x, :, iatom) # TODO SArray?
            append!(types, type[iatom])
            push!(pt, x[:, iatom])
            incut = 1
            for jj in 1:length(neigh)
                jatom = Int(neigh[jj])
                jatom &= NEIGHMASK
                jatom += 1 # 1-based indexing 
                jtype = type[jatom]
                delx = xtmp - x[1, jatom]
                dely = ytmp - x[2, jatom]
                delz = ztmp - x[3, jatom]
                rsq = delx*delx + dely*dely + delz*delz
                if rsq < cutsq
                    append!(types, jtype)
                    push!(pt, x[:, jatom])
                    incut += 1
                end
            end
            fexternal[:, iatom] = compute_force(reshape(pt, (3, incut)), types)[1]
            if eflag
                energies[iatom] = compute_energy(reshape(pt, (3, incut)), types)[1]
            end
        end
        if eflag
            API.lammps_fix_external_set_energy_peratom(fix.lmp, fix.name, energies)
            energy_global!(fix, sum(energies))
        end
        if vflag
            virial = virial_fdotr_compute(fexternal, x, nlocal+nghost)
        end
    end
    FixExternal(pair, lmp, name) 
end
