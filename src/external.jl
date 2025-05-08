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

function _check_atom_datatype(lmp, name, T)
    lammps_type = API.lammps_extract_atom_datatype(lmp, name)
    rows::Int = API.lammps_extract_atom_size(lmp, name, API.LMP_SIZE_COLS)

    if lammps_type == API.LAMMPS_INT64
        T !== Int64 && throw(ArgumentError("Expected type Int64 for per-atom property :$name, got $T instead"))
    elseif lammps_type == API.LAMMPS_INT
        T !== Int32 && throw(ArgumentError("Expected type Int32 for per-atom property :$name, got $T instead"))
    elseif lammps_type == API.LAMMPS_DOUBLE
        T !== Float64 && throw(ArgumentError("Expected type Float64 for per-atom property :$name, got $T instead"))
    elseif lammps_type == API.LAMMPS_INT64_2D
        (T !== NTuple{rows, Int64} && T !== SVector{rows, Int64}) && throw(ArgumentError("Expected type NTuple{$rows, Int64} or SVector{$rows, Int64} for per-atom property :$name, got $T instead"))
    elseif lammps_type == API.LAMMPS_INT_2D
        (T !== NTuple{rows, Int32} && T !== SVector{rows, Int32}) && throw(ArgumentError("Expected type NTuple{$rows, Int32} or SVector{$rows, Int32} for per-atom property :$name, got $T instead"))
    elseif lammps_type == API.LAMMPS_DOUBLE_2D
        (T !== NTuple{rows, Float64} && T !== SVector{rows, Float64}) && throw(ArgumentError("Expected type NTuple{$rows, Float64} or SVector{$rows, Float64} for per-atom property :$name, got $T instead"))
    else
        throw(KeyError(name))
    end
end

function _check_atom_ghost(lmp, name, nall)
    cols = API.lammps_extract_atom_size(lmp, name, API.LMP_SIZE_ROWS)
    cols == nall || throw(ArgumentError("Per-atom property :$name doesn't communicate ghost data"))
    nothing
end

struct ExtractAtomMultiple{T<:NamedTuple, P} <: AbstractVector{T}
    ptrs::P
    length::Int
    
    function ExtractAtomMultiple{T}(lmp) where T
        nall = extract_setting(lmp, "nall")
        ptrs = map(fieldnames(T)) do name
            Base.@constprop :aggressive
            type = fieldtype(T, name)
            void_ptr = API.lammps_extract_atom(lmp, name)
            void_ptr == C_NULL && throw(KeyError(name))
            API.lammps_extract_atom_size(lmp, name, API.LMP_SIZE_ROWS)
            _check_atom_datatype(lmp, name, type)
            _check_atom_ghost(lmp, name, nall)
            if name === :mass
                type_ptr::Ptr{Int32} = API.lammps_extract_atom(lmp, :type)
                (type_ptr, Ptr{type}(void_ptr))
            elseif type<:Real
                Ptr{type}(void_ptr)
            elseif nall == 0
                Ptr{type}(C_NULL)
            else
                unsafe_load(Ptr{Ptr{type}}(void_ptr))
            end
        end
        return new{T, typeof(ptrs)}(ptrs, nall)
    end
end

Base.size(self::ExtractAtomMultiple) = (self.length,)
function Base.getindex(self::ExtractAtomMultiple{T}, i::Integer) where T
    @boundscheck checkbounds(self, i)
    map(self.ptrs) do ptr
        if ptr isa Tuple
            type = unsafe_load(ptr[1])
            return unsafe_load(ptr[2], type+1)
        end
        unsafe_load(ptr, i)
    end |> T
end

function _check_global_datatype(lmp, name, T)
    lammps_type = API.lammps_extract_global_datatype(lmp, name)

    if name in (:sublo, :subhi, :sublo_lambda, :subhi_lambda)
        (T !== NTuple{3, Float64} && T !== SVector{3, Float64}) && throw(ArgumentError("Expected type NTuple{3, Float64} or SVector{3, Float64} for global property :$name, got $T instead"))
    elseif name == :procgrid
        (T !== NTuple{3, Int32} && T !== SVector{3, Int32}) && throw(ArgumentError("Expected type NTuple{3, Int32} or SVector{3, Int32} for global property :$name, got $T instead"))
    elseif name in (:special_lj, :special_coul)
        (T !== NTuple{4, Float64} && T !== SVector{4, Float64}) && throw(ArgumentError("Expected type NTuple{4, Float64} or SVector{4, Float64} for global property :$name, got $T instead"))
    elseif lammps_type == API.LAMMPS_INT64
        T !== Int64 && throw(ArgumentError("Expected type Int64 for global property :$name, got $T instead"))
    elseif lammps_type == API.LAMMPS_INT
        T !== Int32 && throw(ArgumentError("Expected type Int32 for global property :$name, got $T instead"))
    elseif lammps_type == API.LAMMPS_DOUBLE
        T !== Float64 && throw(ArgumentError("Expected type Float64 for global property :$name, got $T instead"))
    elseif lammps_type == API.LAMMPS_STRING
        T !== String && throw(ArgumentError("Expected type String for global property :$name, got $T instead"))
    else
        throw(KeyError(name))
    end
end

struct ExtractGlobalMultiple{T<:NamedTuple, P}
    ptrs::P

    function ExtractGlobalMultiple{T}(lmp) where T
        ptrs = map(fieldnames(T)) do name
            Base.@constprop :aggressive
            type = fieldtype(T, name)
            void_ptr = API.lammps_extract_global(lmp, name)
            void_ptr == C_NULL && throw(KeyError(name))
            _check_global_datatype(lmp, name, type)
            if type === String
                Ptr{Cchar}(void_ptr)
            else
                Ptr{type}(void_ptr)
            end
        end
        return new{T, typeof(ptrs)}(ptrs)
    end
end

function Base.getindex(self::ExtractGlobalMultiple{T}) where T
    map(self.ptrs) do ptr
        ptr isa Ptr{Cchar} ? unsafe_string(ptr) : unsafe_load(ptr)
    end |> T
end

_dott(v) = SA[v.x*v.x, v.y*v.y, v.z*v.z, v.x*v.y, v.x*v.z, v.y*v.z]

function PairExternal(compute_potential::F, lmp::LMP, name::String, system_properties::Type{T}, atom_properties::Type{U}, cutoff::Float64; backend::Union{Nothing, AbstractADType} = nothing) where {F, T<:NamedTuple, U<:NamedTuple}
    command(lmp, """
        pair_style zero $cutoff nocoeff full
        pair_coeff * *
    """)

    system_ptrs = ExtractGlobalMultiple{system_properties}(lmp) # persistent in memory

    FixExternal(lmp, name, "all", 1, 1) do fix::FixExternal
        system = system_ptrs[]
        atom = ExtractAtomMultiple{atom_properties}(lmp)
        x = reinterpret(reshape, SVector{3, Float64}, fix.x)
        force = reinterpret(reshape, SVector{3, Float64}, fix.f)

        @no_escape begin
            energy = @alloc(Float64, fix.nlocal)
            virial = @alloc(SVector{6, Float64}, fix.nlocal)

            @inbounds for (i, neigh) in pair_neighborlist(fix.lmp, "zero")
                ienergy = 0.
                iforce = zero(SVector{3, Float64})
                ivirial = zero(SVector{6, Float64})

                iatom = atom[i]
                ipos = x[i]

                for j in neigh
                    jatom = atom[j]
                    diff = ipos - x[j]
                    r = norm(diff)
                    r > cutoff && continue

                    if backend === nothing
                        _energy, _force,  = compute_potential(r, system, iatom, jatom)
                    else
                        _energy, _derivative = value_and_derivative(compute_potential, backend, r, Constant(system), Constant(iatom), Constant(jatom))
                        _force = -_derivative
                    end

                    ienergy += 0.5 * _energy
                    iforce += diff * (_force / r)
                    ivirial += (0.5 * _force / r) * _dott(diff)
                end

                energy[i] = ienergy
                force[i] = iforce
                virial[i] = ivirial
            end

            set_energy_peratom(fix, energy; set_global=true)
            set_virial_peratom(fix, virial; set_global=true)
        end
    end
end