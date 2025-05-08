function fix_external_callback end

"""
    FixExternal(callback, lmp::LMP, name::String, group::String, ncall::Int, napply::Int)

Creates a fix in lammps that calls a Julia function `callback` every `ncall` during the simulation.

---

The following command is executed in LAMMPS when `FixExternal` is called in order to setup the fix:
```lammps
fix <name> <group> external pf/callback <ncall> <napply>
```

---

The `FixExternal` object gets passed to the `callback` function, it contains the parameters passed to `FixExternal` as fields:
- `lmp::LMP`: The LAMMPS object.
- `name::String`: The name of the fix.
- `group::String`: The group of atoms to which the fix is applied.
- `ncall::Int`: The number of timesteps between calls to the callback function.
- `napply::Int`: The number of times the callback function is applied to the atoms in the group.
Additionally, the following fields are set by the callback function:
- `timestep::Int`: The current timestep of the simulation.
- `nlocal::Int`: The number of local atoms on the MPI rank.
- `ids::UnsafeArray{Int32, 1}`: The IDs of the local atoms in an `nall`-element Vector.
- `x::UnsafeArray{Float64, 2}`: The positions of the local atoms in a (3 × `nall`) Matrix.
- `f::UnsafeArray{Float64, 2}`: The forces on the local atoms in a (3 × `nlocal`) Matrix.
These values are only valid during the callback execution and should *not* be used outside of it, as they refer directly to LAMMPS internal memory.

Here, the intention is for the callback to write forces to the `f` field, which will be applied to the atoms in the group every `napply` timesteps.
`f` is *not* zeroed before the callback is called, so the forces from previous calls are preserved.
"""
mutable struct FixExternal
    const callback
    const lmp::LMP
    const name::String
    const group::String
    const ncall::Int
    const napply::Int
    # The following fields are set by the callback function
    timestep::Int
    nlocal::Int
    ids::UnsafeArray{Int32, 1}
    x::UnsafeArray{Float64, 2}
    f::UnsafeArray{Float64, 2}

    function FixExternal(callback, lmp::LMP, name::String, group::String, ncall::Int, napply::Int)
        command(lmp, "fix $name $group external pf/callback $ncall $napply")
        this = new(callback, lmp, name, group, ncall, napply,  0, 0,
            UnsafeArray(Ptr{Int32}(C_NULL), (0, )),
            UnsafeArray(Ptr{Float64}(C_NULL), (3, 0)), 
            UnsafeArray(Ptr{Float64}(C_NULL), (3, 0))
        )
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
    fix.f = _extract(fexternal, (Int32(3), nlocal))
    fix.ids = _extract(ids, nall)

    # necessary dynamic
    fix.callback(fix)

    fix.timestep = 0
    fix.nlocal = 0

    fix.ids = UnsafeArray(Ptr{Int32}(C_NULL), (0, ))
    fix.x = UnsafeArray(Ptr{Float64}(C_NULL), (3, 0))
    fix.f = UnsafeArray(Ptr{Float64}(C_NULL), (3, 0))

    return nothing
end

"""
    set_virial_peratom(fix::FixExternal, virial::AbstractVector{SVector{6, Float64}}; set_global=false)
    set_virial_peratom(fix::FixExternal, virial::AbstractMatrix{Float64}; set_global=false)

Sets the per-atom virial stress tensor for the atoms in the fix.
The virial tensor is represented as a 6-component symmetric tensor in the order: 
  `[xx, yy, zz, xy, xz, yz]`.
This is intended to only be used during the callback function of a `FixExternal`.
"""
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

function PairExternal(compute_potential::F, lmp::LMP, system_properties::Type{T}, atom_properties::Type{U}, cutoff::Float64; backend::Union{Nothing, AbstractADType} = nothing) where {F, T<:NamedTuple, U<:NamedTuple}
    command(lmp, """
        pair_style zero $cutoff nocoeff
        pair_coeff * *
    """)

    system_ptrs = ExtractGlobalMultiple{system_properties}(lmp) # persistent in memory

    FixExternal(lmp, "pair_julia", "all", 1, 1) do fix::FixExternal
        system = system_ptrs[]
        atom = ExtractAtomMultiple{atom_properties}(lmp)

        @no_escape begin
            x = reinterpret(reshape, SVector{3, Float64}, fix.x)
            e = @alloc(Float64, fix.nlocal)
            f = reinterpret(reshape, SVector{3, Float64}, fix.f)
            v = @alloc(SVector{6, Float64}, fix.nlocal)

            fill!(e, zero(eltype(e)))
            fill!(f, zero(eltype(f)))
            fill!(v, zero(eltype(v)))

            @inbounds for (i, neigh) in pair_neighborlist(fix.lmp, "zero")
                iatom = atom[i]
                ipos = x[i]

                for j in neigh
                    jatom = atom[j]
                    diff = ipos - x[j]
                    r = norm(diff)
                    r > cutoff && continue

                    if backend === nothing
                        energy, force_magnitude = compute_potential(r, system, iatom, jatom)
                    else
                        energy, derivative = value_and_derivative(compute_potential, backend, r, Constant(system), Constant(iatom), Constant(jatom))
                        force_magnitude = -derivative
                    end

                    force = diff * (force_magnitude / r)
                    virial = _dott(diff) * (force_magnitude / r)

                    e[i] += 0.5 * energy
                    f[i] += force
                    v[i] += 0.5 * virial

                    if j <= fix.nlocal
                        e[j] += 0.5 * energy
                        f[j] -= force
                        v[j] += 0.5 * virial
                    end
                end
            end

            set_energy_peratom(fix, e; set_global=true)
            set_virial_peratom(fix, v; set_global=true)
        end
    end
end