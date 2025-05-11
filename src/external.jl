function fix_external_callback end

"""
    FixExternal(callback, lmp::LMP, name::String, group::String, ncall::Int, napply::Int)

Creates a fix in lammps that calls a Julia function `callback` every `ncall` during the simulation.

!!! info "lammps commands"
    The following command is executed in LAMMPS when `FixExternal` is called in order to setup the fix:
    ```lammps
    fix <name> <group> external pf/callback <ncall> <napply>
    ```

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

!!! info "ghost atoms"
    `FixExternal` is invoked at the `post_force` stage of each timestep. Modified data of the ghost atoms will *not* be communicated back to
    the processor owning the respective atom, as at this point the `reverse_comm` stage of the timestep has already happened.

Contributions to the per-atom energies or virials can be set with [`set_energy!`](@ref) and [`set_virial!`](@ref), respectively.

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
    set_virial!(fix::FixExternal, virial_peratom::AbstractVector{SVector{6, Float64}})
    set_virial!(fix::FixExternal, virial_peratom::AbstractMatrix{Float64})

Sets the contribution to the local per-atom and total global virial stress tensors resulting from the `FixExternal`.
This function should only be used within the callback of the `FixExternal`

The virial tensor is represented as a 6-component vector with the order: 
  `[xx, yy, zz, xy, xz, yz]`.
"""
function set_virial!(fix::FixExternal, virial_peratom::AbstractVector{SVector{6, Float64}})
    @assert length(virial_peratom) >= fix.nlocal

    @no_escape begin
        ptrs = @alloc(Ptr{Ptr{Float64}}, fix.nlocal)
        for i in eachindex(ptrs)
            @inbounds ptrs[i] = pointer(virial_peratom, i)
        end
        API.lammps_fix_external_set_virial_peratom(fix.lmp, fix.name, ptrs)
    end

    virial_global = MVector(sum(view(virial_peratom, 1:fix.nlocal)))
    comm = get_mpi_comm(fix.lmp)
    GC.@preserve virial_global begin
        if comm !== nothing
            buffer = MPI.RBuffer(MPI.IN_PLACE, pointer(virial_global), 6, MPI.Datatype(Float64))
            @inline MPI.Allreduce!(buffer, +, comm)
        end
        @inline API.lammps_fix_external_set_virial_global(fix.lmp, fix.name, virial_global)
    end

    check(fix.lmp)
end

function set_virial(fix::FixExternal, virial::AbstractMatrix{Float64})
    virial_vec = reinterpret(reshape, SVector{6, Float64}, virial)
    set_virial_peratom(fix, virial_vec; set_global)
end

"""
    set_energy!(fix::FixExternal, energy::AbstractVector{Float64}; set_global=false)

Sets the contribution to the local per-atom and total global energy resulting from the `FixExternal`.
This function should only be used within the callback of the `FixExternal`
"""
function set_energy!(fix::FixExternal, energy::AbstractVector{Float64})
    @assert length(energy) >= fix.nlocal

    API.lammps_fix_external_set_energy_peratom(fix.lmp, fix.name, energy)

    energy_global = sum(view(energy, 1:fix.nlocal))
    comm = get_mpi_comm(fix.lmp)
    if comm !== nothing
        energy_global = MPI.Allreduce(energy_global, +, comm)
    end
    API.lammps_fix_external_set_energy_global(fix.lmp, fix.name, energy_global)

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

"""
    InteractionConfig(;
        system::Type{<:NamedTuple} = @NamedTuple{},
        atom::Type{<:NamedTuple} = @NamedTuple{type::Int32},
        backend::Union{Nothing, AbstractADType} = nothing
    )

A configuration struct that encapsulates the system properties, atom properties, and an optional differentiation backend for use in external pair interactions.

# Fields
- `system`: A `NamedTuple` type that defines the global system properties to be extracted from LAMMPS. Defaults to an empty `NamedTuple`.
    A list of the available global properties can be found in the [LAMMPS documentation](https://docs.lammps.org/Library_properties.html#_CPPv421lammps_extract_globalPvPKc)
- `atom`: A `NamedTuple` type that defines the per-atom properties to be extracted from LAMMPS. Defaults to `@NamedTuple{type::Int32}`.
    A list of the available atom properties can be found in the [LAMMPS documentation](https://docs.lammps.org/Classes_atom.html#_CPPv4N9LAMMPS_NS4Atom7extractEPKc)
- `backend`: An optional automatic differentiation backend used to automatically calculate interaction forces from a potential function. Defaults to `nothing`.
    The backends are defined through the [`ADTypes` package](https://docs.sciml.ai/ADTypes/stable/).

# Example
```julia
config = InteractionConfig(
    system = @NamedTuple{qqrd2e::Float64, boltz::Float64},
    atom = @NamedTuple{type::Int32, q::Float64},
    backend = nothing
)
```
"""
struct InteractionConfig{T<:NamedTuple, U<:NamedTuple, B<:Union{Nothing, AbstractADType}}
    backend::B

    function InteractionConfig(;
            system::Type{T} = @NamedTuple{},
            atom::Type{U} = @NamedTuple{type::Int32},
            backend::B = nothing,
        ) where {T<:NamedTuple, U<:NamedTuple, B<:Union{Nothing, AbstractADType}}
        return new{T, U, B}(backend)
    end
end

_dott(v) = SA[v.x*v.x, v.y*v.y, v.z*v.z, v.x*v.y, v.x*v.z, v.y*v.z]
_dot(v, u) = SA[v.x*u.x, v.y*u.y, v.z*u.z, v.x*u.y, v.x*u.z, v.y*u.z]

"""
    PairExternal(compute_potential, lmp::LMP, config::InteractionConfig, cutoff::Float64)

Defines a custom pair style in LAMMPS using a user-provided potential function.

!!! warning "limitations"
    `PairExternal` is implemented as a `FixExternal` and only works with `newton off`.
    Fixes that operate on forces, energies, or virials should also be defined *after* calling `PairExternal`.
    Furthermore, it is not possible to define multiple `PairExternals` for a simulation. Attempting to do this
    will overwrite the previously defined `PairExternal`. 

!!! info "lammps commands"
    The following commands are executed in LAMMPS during the setup process of `PairExternal`:
    ```lammps
    fix pair_julia all external pf/callback 1 1
    pair_style zero <cutoff> nocoeff
    pair_coeff * *
    newton off <newton_bond> # turns off newton_pair while leaving newton_bond unchanged
    ```

`compute_potential` should have the following signature:
```julia
function compute_potential(r::Real, system::config.system, iatom::config.atom, jatom::config.atom)
```
If no differentiation backend is provided through `config`, the function should return a tuple of the pair energy and force magnitude.
Otherwise it should return only the energy.

## Examples
```julia
# without automatic differentiation
config = InteractionConfig(
    system = @NamedTuple{qqrd2e::Float64},
    atom = @NamedTuple{q::Float64},
)

PairExternal(lmp, config, 2.5) do r, system, iatom, jatom
    energy = system.qqrd2e * (iatom.q * jatom.q) / r
    force = system.qqrd2e * (iatom.q * jatom.q) / r^2
    return energy, force
end

# with automatic differentiation
import ADTypes: AutoEnzyme()
import Enzyme

config = InteractionConfig(
    system = @NamedTuple{qqrd2e::Float64},
    atom = @NamedTuple{q::Float64},
    backend = AutoEnzyme(),
)

PairExternal(lmp, config, 2.5) do r, system, iatom, jatom
    system.qqrd2e * (iatom.q * jatom.q) / r
end
```

"""
function PairExternal(compute_potential::F, lmp::LMP, config::InteractionConfig{T, U}, cutoff::Float64) where {F, T, U}
    newton_bond = extract_setting(lmp, "newton_bond") == 0 ? "off" : "on"
    command(lmp, """
        pair_style zero $cutoff nocoeff
        pair_coeff * *
        newton off $newton_bond
    """)

    system_ptrs = ExtractGlobalMultiple{T}(lmp) # persistent in memory

    FixExternal(lmp, "pair_julia", "all", 1, 1) do fix::FixExternal
        @assert extract_setting(lmp, "newton_pair") == 0
        system = system_ptrs[]
        atom = ExtractAtomMultiple{U}(lmp)

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

                    if config.backend === nothing
                        energy, force_magnitude = compute_potential(r, system, iatom, jatom)
                    else
                        energy, derivative = value_and_derivative(compute_potential, config.backend, r, Constant(system), Constant(iatom), Constant(jatom))
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

            set_energy!(fix, e)
            set_virial!(fix, v)
        end
    end
end

function BondExternal(compute_potential::F, lmp::LMP, config::InteractionConfig{T, U}) where {F, T, U}
    command(lmp, """
        bond_style zero
        bond_coeff *
    """)

    system_ptrs = ExtractGlobalMultiple{T}(lmp) # persistent in memory

    FixExternal(lmp, "bond_julia", "all", 1, 1) do fix::FixExternal
        @assert extract_setting(lmp, "newton_bond") == 0
        system = system_ptrs[]
        atom = ExtractAtomMultiple{U}(lmp)

        @no_escape begin
            x = reinterpret(reshape, SVector{3, Float64}, fix.x)
            e = @alloc(Float64, fix.nlocal)
            f = reinterpret(reshape, SVector{3, Float64}, fix.f)
            v = @alloc(SVector{6, Float64}, fix.nlocal)

            fill!(e, zero(eltype(e)))
            fill!(f, zero(eltype(f)))
            fill!(v, zero(eltype(v)))

            @inbounds for (i, j, type) in bond_neighborlist(lmp)
                diff = x[i] - x[j]
                r = norm(diff)
                iatom = atom[i]
                jatom = atom[j]

                if config.backend === nothing
                    energy, force_magnitude = compute_potential(r, system, type, iatom, jatom)
                else
                    energy, derivative = value_and_derivative(compute_potential, config.backend, r, Constant(type), Constant(system), Constant(iatom), Constant(jatom))
                    force_magnitude = -derivative
                end

                force = diff * (force_magnitude / r)
                virial = _dott(diff) * (force_magnitude / r)

                if i <= fix.nlocal
                    e[i] += 0.5 * energy
                    f[i] += force
                    v[i] += 0.5 * virial
                end

                if j <= fix.nlocal
                    e[j] += 0.5 * energy
                    f[j] -= force
                    v[j] += 0.5 * virial
                end
            end

            set_energy!(fix, e)
            set_virial!(fix, v)
        end
    end
end

function AngleExternal(compute_potential::F, lmp::LMP, config::InteractionConfig{T, U}) where {F, T, U}
    command(lmp, """
        angle_style zero
        angle_coeff *
    """)

    system_ptrs = ExtractGlobalMultiple{T}(lmp) # persistent in memory

    FixExternal(lmp, "angle_julia", "all", 1, 1) do fix::FixExternal
        @assert extract_setting(lmp, "newton_bond") == 0
        system = system_ptrs[]
        atom = ExtractAtomMultiple{U}(lmp)

        @no_escape begin
            x = reinterpret(reshape, SVector{3, Float64}, fix.x)
            e = @alloc(Float64, fix.nlocal)
            f = reinterpret(reshape, SVector{3, Float64}, fix.f)
            v = @alloc(SVector{6, Float64}, fix.nlocal)

            fill!(e, zero(eltype(e)))
            fill!(f, zero(eltype(f)))
            fill!(v, zero(eltype(v)))

            @inbounds for (i, j, k, type) in angle_neighborlist(lmp)
                diff1 = x[i] - x[j]
                diff2 = x[k] - x[j]
                r1 = norm(diff1)
                r2 = norm(diff2)

                c = (diff1 ⋅ diff2)/(r1 * r2)
                c = clamp(0, c, 1)
                s = 1 / sqrt(1 - c^2)
                θ = acos(c)

                iatom = atom[i]
                jatom = atom[j]
                katom = atom[k]
                energy, derivative = value_and_derivative(compute_potential, config.backend, θ, Constant(type), Constant(system), Constant(iatom), Constant(jatom), Constant(katom))

                a = - derivative * s
                a11 = a * c / (r1 * r1)
                a12 = - a / (r1 * r2)
                a22 = a * c / (r2 * r2)

                f1 = a11 * diff1 + a12 * diff2
                f3 = a22 * diff2 + a12 * diff1

                virial = _dot(diff1, f1) + _dot(diff2, f3)

                if i <= fix.nlocal
                    e[i] += energy/3
                    f[i] += f1
                    v[i] += virial/3
                end

                if j <= fix.nlocal
                    e[j] += energy/3
                    f[j] -= f1 + f3
                    v[j] += virial/3
                end

                if k <= fix.nlocal
                    e[k] += energy/3
                    f[k] += f3
                    v[k] += virial/3
                end
            end

            set_energy!(fix, e)
            set_virial!(fix, v)
        end
    end
end