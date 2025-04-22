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
    dim = extract_setting(fix.lmp, "dimension")
    fix.x = _extract(x, (dim, nall))
    fix.f = _extract(fexternal, (dim, nall))
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

function PairExternal(compute_potential::F, lmp::LMP, name::String, cutoff::Float64) where F
    command(lmp, """
        pair_style zero $cutoff nocoeff full
        pair_coeff * *
        fix $(name)_intermediate all property/atom d_$(name)_energy d2_$(name)_virial 6
    """)

    FixExternal(lmp, name, "all", 1, 1) do fix::FixExternal
        idx = API.lammps_find_pair_neighlist(fix.lmp, "zero", true, 0, 0)
        nelements = API.lammps_neighlist_num_elements(fix.lmp, idx)

        type = LAMMPS.extract_atom(fix.lmp, "type", LAMMPS_INT; with_ghosts=true)
        x = reinterpret(Vec{3, Float64}, fix.x)
        force = reinterpret(Vec{3, Float64}, fix.f)
        energy = extract_atom(fix.lmp, "d_$(fix.name)_energy", LAMMPS_DOUBLE)

        virial_void_ptr = API.lammps_extract_atom(fix.lmp, "d2_$(fix.name)_virial")
        virial_ptr = reinterpret(Ptr{Ptr{Float64}}, virial_void_ptr)
        virial = _extract(virial_ptr, (Int32(6), nelements))

        energy_tally = 0.
        virial_tally = zero(SymmetricTensor{2,3,Float64,6})

        energy .= 0
        fix.f .= 0
        virial .= 0

        for i in 1:nelements
            iatom, neigh = LAMMPS.neighbors(lmp, idx, i)
            iatom += 1
            itype = type[iatom]
            ipos = x[iatom]

            for jatom in neigh
                jatom += 1
                jtype = type[jatom]
                diff = x[jatom] - ipos
                r = norm(diff)
                r > cutoff && continue

                _force, _energy = gradient(r, :all) do r
                    compute_potential(r, itype, jtype)
                end

                v = diff / r

                energy[iatom] += 0.5 * _energy
                force[iatom] += diff * (_force / r)

                virial_tens = (-0.5 * _force / r) * symmetric(diff ⊗ diff)
                virial[1, iatom] += virial_tens[1,1]
                virial[2, iatom] += virial_tens[2,2]
                virial[3, iatom] += virial_tens[3,3]
                virial[4, iatom] += virial_tens[1,2]
                virial[5, iatom] += virial_tens[1,3]
                virial[6, iatom] += virial_tens[2,3]

                virial_tally += virial_tens
                energy_tally += 0.5 * _energy
            end
        end

        API.lammps_fix_external_set_energy_peratom(fix.lmp, fix.name, energy)
        API.lammps_fix_external_set_virial_peratom(fix.lmp, fix.name, virial_ptr)

        energy_tally = MPI.Allreduce(energy_tally, +, fix.lmp.comm)
        virial_tally = SymmetricTensor{2,3,Float64,6}(MPI.Allreduce(virial_tally.data, +, fix.lmp.comm))

        API.lammps_fix_external_set_energy_global(fix.lmp, fix.name, energy_tally)
        virial_tally2 = Ref((virial_tally[1,1], virial_tally[2,2], virial_tally[3,3], virial_tally[1,2], virial_tally[1,3], virial_tally[2,3]))
        API.lammps_fix_external_set_virial_global(fix.lmp, fix.name, virial_tally2)
    end
end