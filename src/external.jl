function fix_external_callback end

mutable struct FixExternal
    const lmp::LMP
    const name::String
    const callback
    timestep::Int
    nlocal::Int
    ids::Union{Nothing, Vector{Cint}}
    x::Union{Nothing, Matrix{Float64}}
    f::Union{Nothing, Matrix{Float64}}

    function FixExternal(callback, lmp::LMP, name::String, group::String, ncall::Int, napply::Int)
        command(lmp, "fix $name $group external pf/callback $ncall $napply")
        this = new(lmp, name, callback, 0, 0, nothing, nothing, nothing)
        lmp.external_fixes[name] = this # preserves pair globally

        ctx = Base.pointer_from_objref(this)
        callback = @cfunction(fix_external_callback, Cvoid, (Ptr{Cvoid}, Int64, Cint, Ptr{Cint}, Ptr{Ptr{Float64}}, Ptr{Ptr{Float64}}))
        API.lammps_set_fix_external_callback(lmp, name, callback, ctx)

        # Ensure function is compiled before timestep 0
        if !precompile(this.callback, (FixExternal,))
            @warn "Failed to precompile the callback" this.callback
        end
        return this
    end
end

function fix_external_callback(ctx::Ptr{Cvoid}, timestep::Int64, nlocal::Cint, ids::Ptr{Cint}, x::Ptr{Ptr{Float64}}, fexternal::Ptr{Ptr{Float64}})
    fix = Base.unsafe_pointer_to_objref(ctx)::FixExternal
    fix.timestep = timestep
    fix.nlocal = nlocal

    nall = extract_setting(fix.lmp, "nall")
    fix.x = _extract(x, (Int32(3), nall))
    fix.f = _extract(fexternal, (Int32(3), nall))
    fix.ids = _extract(ids, nall)

    # necessary dynamic
    fix.callback(fix)

    fix.timestep = 0
    fix.nlocal = 0
    fix.x = nothing
    fix.f = nothing
    fix.ids = nothing

    return nothing
end

function set_virial_peratom(fix::FixExternal, virial; set_global=false)
    @assert eltype(virial) == Float64
    @assert size(virial, 2) >= fix.nlocal
    @assert size(virial, 1) == 6 
    @assert stride(virial, 1) == 1

    @no_escape begin
        ptrs = @alloc(Ptr{Ptr{Float64}}, fix.nlocal)
        @inbounds for i in eachindex(ptrs)
            ptrs[i] = pointer(virial, 6*(i-1)+1)
        end
        API.lammps_fix_external_set_virial_peratom(fix.lmp, fix.name, ptrs)
    end

    @no_escape if set_global
        virial_global = @alloc(Float64, 6)
        virial_global .= 0
        for i in 1:fix.nlocal, j in 1:6
            @inbounds virial_global[j] += virial[6*(i-1) + j]
        end

        comm = API.lammps_get_mpi_comm(fix.lmp)
        if comm != -1
            MPI.Allreduce!(virial_global, +, MPI.Comm(comm))
        end

        API.lammps_fix_external_set_virial_global(fix.lmp, fix.name, virial_global)
    end

    check(fix.lmp)
end

function set_energy_peratom(fix::FixExternal, energy; set_global=false)
    @assert length(energy) >= fix.nlocal

    API.lammps_fix_external_set_energy_peratom(fix.lmp, fix.name, energy)

    if set_global
        energy_global = sum(view(energy, 1:fix.nlocal))
        comm = API.lammps_get_mpi_comm(fix.lmp)
        if comm != -1
            energy_global = MPI.Allreduce(energy_global, +, MPI.Comm(comm))
        end
        API.lammps_fix_external_set_energy_global(fix.lmp, fix.name, energy_global)
    end

    check(fix.lmp)
end

function PairExternal(compute_potential::F, lmp::LMP, name::String, cutoff::Float64) where F
    command(lmp, """
        pair_style zero $cutoff nocoeff full
        pair_coeff * *
    """)

    FixExternal(lmp, name, "all", 1, 1) do fix::FixExternal
        idx = API.lammps_find_pair_neighlist(fix.lmp, "zero", true, 0, 0)
        nelements = API.lammps_neighlist_num_elements(fix.lmp, idx)

        type = LAMMPS.extract_atom(fix.lmp, "type", LAMMPS_INT; with_ghosts=true)
        x = reinterpret(reshape, Vec{3, Float64}, fix.x)
        force = reinterpret(reshape, Vec{3, Float64}, fix.f)

        @no_escape begin
            energy = @alloc(Float64, fix.nlocal)
            virial = @alloc(Float64, 6, fix.nlocal)

            @inbounds for (iatom, neigh) in pair_neighborlist(fix.lmp, "zero")
                itype = type[iatom]
                ipos = x[iatom]
    
                for jatom in neigh
                    jtype = type[jatom]
                    diff = x[jatom] - ipos
                    r = norm(diff)
                    r > cutoff && continue
    
                    _force, _energy = gradient(r, :all) do r
                        compute_potential(r, itype, jtype)
                    end
    
                    energy[iatom] += 0.5 * _energy
                    force[iatom] += diff * (_force / r)
    
                    virial_tens = (-0.5 * _force / r) * symmetric(diff ⊗ diff)
                    virial[1, iatom] += virial_tens[1,1]
                    virial[2, iatom] += virial_tens[2,2]
                    virial[3, iatom] += virial_tens[3,3]
                    virial[4, iatom] += virial_tens[1,2]
                    virial[5, iatom] += virial_tens[1,3]
                    virial[6, iatom] += virial_tens[2,3]
                end
            end

            set_energy_peratom(fix, energy; set_global=true)
            set_virial_peratom(fix, virial; set_global=true)
        end
    end
end