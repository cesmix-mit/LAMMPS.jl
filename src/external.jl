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

function set_virial_peratom(fix::FixExternal, virial::AbstractVector{SVector{6, Float64}}; set_global=false)
    @assert length(virial) >= fix.nlocal

    @no_escape begin
        ptrs = @alloc(Ptr{Ptr{Float64}}, fix.nlocal)
        for i in eachindex(ptrs)
            @inbounds ptrs[i] = pointer(virial, i)
        end
        API.lammps_fix_external_set_virial_peratom(fix.lmp, fix.name, ptrs)
    end

    if set_global
        virial_global = MVector(sum(view(virial, 1:fix.nlocal)))
        comm = get_mpi_comm(fix.lmp)
        GC.@preserve virial_global begin
            if comm !== nothing
                buffer = MPI.RBuffer(MPI.IN_PLACE, pointer(virial_global), 6, MPI.Datatype(Float64))
                @inline MPI.Allreduce!(buffer, +, comm)
            end
            @inline API.lammps_fix_external_set_virial_global(fix.lmp, fix.name, virial_global)
        end
    end

    check(fix.lmp)
end

function set_virial_peratom(fix::FixExternal, virial::AbstractMatrix{Float64}; set_global=false)
    virial_vec = reinterpret(reshape, SVector{6, Float64}, virial)
    set_virial_peratom(fix, virial_vec; set_global)
end

function set_energy_peratom(fix::FixExternal, energy; set_global=false)
    @assert length(energy) >= fix.nlocal

    API.lammps_fix_external_set_energy_peratom(fix.lmp, fix.name, energy)

    if set_global
        energy_global = sum(view(energy, 1:fix.nlocal))
        comm = get_mpi_comm(fix.lmp)
        if comm !== nothing
            energy_global = MPI.Allreduce(energy_global, +, comm)
        end
        API.lammps_fix_external_set_energy_global(fix.lmp, fix.name, energy_global)
    end

    check(fix.lmp)
end

_dott(v) = SA[v.x*v.x, v.y*v.y, v.z*v.z, v.x*v.y, v.x*v.z, v.y*v.z]

function PairExternal(compute_potential::F, lmp::LMP, name::String, cutoff::Float64; backend::Union{Nothing, AbstractADType} = nothing) where F
    command(lmp, """
        pair_style zero $cutoff nocoeff full
        pair_coeff * *
    """)

    FixExternal(lmp, name, "all", 1, 1) do fix::FixExternal
        type = LAMMPS.extract_atom(fix.lmp, "type", LAMMPS_INT; with_ghosts=true)
        x = reinterpret(reshape, SVector{3, Float64}, fix.x)
        force = reinterpret(reshape, SVector{3, Float64}, fix.f)

        @no_escape begin
            energy = @alloc(Float64, fix.nlocal)
            virial = @alloc(SVector{6, Float64}, fix.nlocal)

            @inbounds for (iatom, neigh) in pair_neighborlist(fix.lmp, "zero")
                ienergy = 0.
                iforce = zero(SVector{3, Float64})
                ivirial = zero(SVector{6, Float64})

                itype = type[iatom]
                ipos = x[iatom]

                for jatom in neigh
                    jtype = type[jatom]
                    diff = ipos - x[jatom]
                    r = norm(diff)
                    r > cutoff && continue

                    if backend === nothing
                        _energy, _force,  = compute_potential(r, itype, jtype)
                    else
                        _energy, _derivative = value_and_derivative(compute_potential, backend, r, Constant(itype), Constant(jtype))
                        _force = -_derivative
                    end

                    ienergy += 0.5 * _energy
                    iforce += diff * (_force / r)
                    ivirial += (0.5 * _force / r) * _dott(diff)
                end

                energy[iatom] = ienergy
                force[iatom] = iforce
                virial[iatom] = ivirial
            end

            set_energy_peratom(fix, energy; set_global=true)
            set_virial_peratom(fix, virial; set_global=true)
        end
    end
end