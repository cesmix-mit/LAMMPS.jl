function _get_count(lmp::LMP, name::String)
    # values taken from: https://docs.lammps.org/Classes_atom.html#_CPPv4N9LAMMPS_NS4Atom7extractEPKc

    if startswith(name, r"[f,c]_")
        if name[1] == 'c'
            API.lammps_has_id(lmp, "compute", name[3:end]) != 1 && error("Unknown per atom compute $name")
            count_ptr = API.lammps_extract_compute(lmp::LMP, name[3:end], API.LMP_STYLE_ATOM, API.LMP_SIZE_COLS)
        else
            API.lammps_has_id(lmp, "fix", name[3:end]) != 1 && error("Unknown per atom fix $name")
            count_ptr = API.lammps_extract_fix(lmp::LMP, name[3:end], API.LMP_STYLE_ATOM, API.LMP_SIZE_COLS, 0, 0)
        end
        check(lmp)

        count_ptr = reinterpret(Ptr{Cint}, count_ptr)
        count = unsafe_load(count_ptr)
    
        # a count of 0 indicates that the entity is a vector. In order to perserve type stability we just treat that as a 1xN Matrix.
        return count == 0 ? Cint(1) : count
    else
        type = API.lammps_extract_atom_datatype(lmp, name)
        type == -1 && error("Unknown per-atom property $name")
        if type in (API.LAMMPS_INT_2D, API.LAMMPS_DOUBLE_2D, API.LAMMPS_INT64_2D)
            API.lammps_extract_atom_size(lmp, name, API.LMP_SIZE_COLS)
        else
            return Cint(1)
        end
    end
end

function _get_T(lmp::LMP, name::String)
    if startswith(name, r"[f,c]_")
        return Float64 # computes and fixes are allways doubles
    end

    type = API.lammps_extract_atom_datatype(lmp, name)
    check(lmp)

    if type in (API.LAMMPS_INT, API.LAMMPS_INT_2D)
        return Int32
    elseif type in (API.LAMMPS_DOUBLE, API.LAMMPS_DOUBLE_2D)
        return Float64
    else
        error("Unkown per atom property $name")
    end
end

function _gather!(lmp::LMP, name::String, data::AbstractMatrix{T}, ids, count, natoms, ndata) where {T <: Union{Int32, Float64}}

    name == "mass" && error("scattering/gathering mass is currently not supported! Use `extract_atom()` instead.")
    
    _T = _get_T(lmp, name)

    @assert ismissing(_T) || _T == T "Expected data type $_T got $T instead."

    dtype = (T === Float64)

     if isnothing(ids)
        API.lammps_gather(lmp, name, dtype, count, data)
    else
        @assert all(1 <= id <= natoms for id in ids)
        API.lammps_gather_subset(lmp, name, dtype, count, ndata, ids, data)
    end

    check(lmp)
    return data

end

"""
    gather(lmp::LMP, name::String, T::Union{Type{Int32}, Type{Float64}}, ids::Union{Nothing, Array{Int32}}=nothing)

Gather the named per-atom, per-atom fix, per-atom compute, or fix property/atom-based entities from all processes.
By default (when `ids=nothing`), this method collects data from all atoms in consecutive order according to their IDs.
The optional parameter `ids` determines for which subset of atoms the requested data will be gathered. The returned data will then be ordered according to `ids`

Compute entities have the prefix `c_`, fix entities use the prefix `f_`, and per-atom entites have no prefix.

The returned Array is decoupled from the internal state of the LAMMPS instance.

!!! warning "ids"
    The optional parameter `ids` only works, if there is a map defined. For example by doing:
    `command(lmp, "atom_modify map yes")`
    However, LAMMPS only issues a warning if that's the case, which unfortuately cannot be detected through the underlying API.
    Starting form LAMMPS version `17 Apr 2024` this should no longer be an issue, as LAMMPS then throws an error instead of a warning.
"""
function gather(lmp::LMP, name::String, T::Union{Type{Int32}, Type{Float64}}, ids::Union{Nothing, Array{Int32}}=nothing)

    count = _get_count(lmp, name)
    natoms = get_natoms(lmp)
    ndata = isnothing(ids) ? natoms : length(ids)

    data = Matrix{T}(undef, (count, ndata))

   return _gather!(lmp, name, data, ids, count, natoms, ndata)

end

"""
    gather!(lmp::LMP, name::String, data::AbstractMatrix{T}, ids::Union{Nothing, Array{Int32}}=nothing)

Gather the named per-atom, per-atom fix, per-atom compute, or fix property/atom-based entities from all processes and store the result in data.
By default (when `ids=nothing`), this method collects data from all atoms in consecutive order according to their IDs.
The optional parameter `ids` determines for which subset of atoms the requested data will be gathered. The returned data will then be ordered according to `ids`

Compute entities have the prefix `c_`, fix entities use the prefix `f_`, and per-atom entites have no prefix.

The returned Array is decoupled from the internal state of the LAMMPS instance.

!!! warning "ids"
    The optional parameter `ids` only works, if there is a map defined. For example by doing:
    `command(lmp, "atom_modify map yes")`
    However, LAMMPS only issues a warning if that's the case, which unfortuately cannot be detected through the underlying API.
    Starting form LAMMPS version `17 Apr 2024` this should no longer be an issue, as LAMMPS then throws an error instead of a warning.
"""
function gather!(lmp::LMP, name::String, data::AbstractMatrix{T}, ids::Union{Nothing, Array{Int32}}=nothing) where {T <: Union{Int32, Float64}}
    
    count = _get_count(lmp, name)
    natoms = get_natoms(lmp)
    ndata = isnothing(ids) ? natoms : length(ids)

    if !_array_stride_valid(data)
        throw(ArgumentError("data must be contiguous in memory (i.e., interpretable as a 1D array)"))
    end

    if size(data) != (count, ndata)
        throw(ArgumentError("Dimension of provided storage must be $(count) x $(ndata) for name $name. Got $(size(data))"))
    end

    return _gather!(lmp, name, data, ids, count, natoms, ndata)

end

"""
    scatter!(lmp::LMP, name::String, data::AbstractVecOrMat{T}, ids::Union{Nothing, Array{Int32}}=nothing) where T<:Union{Int32, Float64}

Scatter the named per-atom, per-atom fix, per-atom compute, or fix property/atom-based entity in data to all processes.
By default (when `ids=nothing`), this method scatters data to all atoms in consecutive order according to their IDs.
The optional parameter `ids` determines to which subset of atoms the data will be scattered.

Compute entities have the prefix `c_`, fix entities use the prefix `f_`, and per-atom entites have no prefix.

!!! warning "ids"
    The optional parameter `ids` only works, if there is a map defined. For example by doing:
    `command(lmp, "atom_modify map yes")`
    However, LAMMPS only issues a warning if that's the case, which unfortuately cannot be detected through the underlying API.
    Starting form LAMMPS version `17 Apr 2024` this should no longer be an issue, as LAMMPS then throws an error instead of a warning.
"""
function scatter!(lmp::LMP, name::String, data::AbstractVecOrMat{T}, ids::Union{Nothing, Array{Int32}}=nothing) where T<:Union{Int32, Float64}
    name == "mass" && error("scattering/gathering mass is currently not supported! Use `extract_atom()` instead.")

    if !_array_stride_valid(data)
        throw(ArgumentError("data must be contiguous in memory (i.e., interpretable as a 1D array)"))
    end

    count = _get_count(lmp, name)
    _T = _get_T(lmp, name)

    @assert ismissing(_T) || _T == T "Expected data type $_T got $T instead."

    dtype = (T === Float64)
    natoms = get_natoms(lmp)
    ndata = isnothing(ids) ? natoms : length(ids)

    if data isa Vector
        @assert count == 1
        @assert ndata == length(data)
    else
        @assert count == size(data,1)
        @assert ndata == size(data,2)
    end

    if isnothing(ids)
        API.lammps_scatter(lmp, name, dtype, count, data)
    else
        @assert all(1 <= id <= natoms for id in ids)
        API.lammps_scatter_subset(lmp, name, dtype, count, ndata, ids, data)
    end

    check(lmp)
end

"""
    gather_bonds(lmp::LMP)

Gather the list of all bonds into a 3 x nbonds Matrix:
```
row1 -> bond type
row2 -> atom 1
row3 -> atom 2
```
"""
function gather_bonds(lmp::LMP)
    ndata = extract_global(lmp, "nbonds", LAMMPS_INT64)[]
    data = Matrix{Int32}(undef, 3, ndata)
    API.lammps_gather_bonds(lmp, data)
    return data
end

"""
    gather_angles(lmp::LMP)

Gather the list of all angles into a 4 x nangles Matrix:
```
row1 -> angle type
row2 -> atom 1
row3 -> atom 2
row4 -> atom 3
```
"""
function gather_angles(lmp::LMP)
    ndata = extract_global(lmp, "nangles", LAMMPS_INT64)[]
    data = Matrix{Int32}(undef, 4, ndata)
    API.lammps_gather_angles(lmp, data)
    return data
end

"""
    gather_dihedrals(lmp::LMP)

Gather the list of all dihedrals into a 5 x ndihedrals Matrix:
```
row1 -> dihedral type
row2 -> atom 1
row3 -> atom 2
row4 -> atom 3
row5 -> atom 4
```
"""
function gather_dihedrals(lmp::LMP)
    ndata = extract_global(lmp, "ndihedrals", LAMMPS_INT64)[]
    data = Matrix{Int32}(undef, 5, ndata)
    API.lammps_gather_dihedrals(lmp, data)
    return data
end

"""
    gather_impropers(lmp::LMP)

Gather the list of all impropers into a 5 x nimpropers Matrix:
```
row1 -> improper type
row2 -> atom 1
row3 -> atom 2
row4 -> atom 3
row5 -> atom 4
```
"""
function gather_impropers(lmp::LMP)
    ndata = extract_global(lmp, "nimpropers", LAMMPS_INT64)[]
    data = Matrix{Int32}(undef, 5, ndata)
    API.lammps_gather_impropers(lmp, data)
    return data
end

"""
    create_atoms(
        lmp::LMP, x::AbstractMatrix{Float64}, id::Vector{Int32}, types::Vector{Int32};
        v::Union{Nothing,Matrix{Float64}}=nothing,
        image::Union{Nothing,Vector{IMAGEINT}}=nothing,
        bexpand::Bool=false
    )

Create atoms for a LAMMPS instance. 
`x` contains the atom positions and should be a 3 by `n` `Matrix{Float64}`, where `n` is the number of atoms. 
`id` contains the id of each atom and should be a length `n` `Vector{Int32}`.
`types` contains the atomic type (LAMMPS number) of each atom and should be a length `n` `Vector{Int32}`.
`v` contains the associated velocities and should be a 3 by `n` `Matrix{Float64}`.
`image` contains the image flags for each atom and should be a length `n` `Vector{IMAGEINT}`.
`bexpand` is a `Bool` that defines whether or not the box should be expanded to fit the input atoms (default not).
"""
function create_atoms(
    lmp::LMP, x::AbstractMatrix{Float64}, id::Vector{Int32}, types::Vector{Int32};
    v::Union{Nothing,Matrix{Float64}}=nothing,
    image::Union{Nothing,Vector{IMAGEINT}}=nothing,
    bexpand::Bool=false
)
    numAtoms = size(x, 2)
    if !_array_stride_valid(x)
        throw(ArgumentError("x must be contiguous in memory (i.e., interpretable as a 1D array)"))
    end
    if size(x, 1) != 3
        throw(ArgumentError("x must be a n by 3 matrix, where n is the number of atoms"))
    end
    if numAtoms != length(id)
        throw(ArgumentError("id must have the same length as the number of atoms"))
    end
    if numAtoms != length(types)
        throw(ArgumentError("types must have the same length as the number of atoms"))
    end
    if v != nothing && size(x) != size(v)
        throw(ArgumentError("x and v must be the same size"))
    end
    if image != nothing && numAtoms != length(image)
        throw(ArgumentError("image must have the same length as the number of atoms"))
    end

    v = v == nothing ? C_NULL : v
    image = image == nothing ? C_NULL : image

    API.lammps_create_atoms(lmp, numAtoms, id, types, x, v, image, bexpand ? 1 : 0)
end
