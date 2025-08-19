function _get_type_and_count(lmp::LMP, name::String, decode_image::Bool)
    count::Int = 0
    type::Int = -1

    if startswith(name, r"[f,c]_")
        actual_name = @view name[3:end]
        count_ptr::Ptr{Cint} = name[1] == 'c' ?
            API.lammps_extract_compute(lmp::LMP, actual_name, API.LMP_STYLE_ATOM, API.LMP_SIZE_COLS) :
            API.lammps_extract_fix(lmp::LMP, actual_name, API.LMP_STYLE_ATOM, API.LMP_SIZE_COLS, 0, 0)
        check(lmp)
        count = unsafe_load(count_ptr)
        type = iszero(count) ? API.LAMMPS_DOUBLE : API.LAMMPS_DOUBLE_2D
    elseif decode_image && name == "image"
        type = API.LAMMPS_INT_2D
        count = 3
    else
        type = API.lammps_extract_atom_datatype(lmp, name)
        type == -1 && throw(ArgumentError("Unknown per-atom property $name"))
        if type in (API.LAMMPS_INT_2D, API.LAMMPS_DOUBLE_2D, API.LAMMPS_INT64_2D)
            count = API.lammps_extract_atom_size(lmp, name, API.LMP_SIZE_COLS)
        end
    end
    return type, count
end

function _check_array(lmp::LMP, name::String, data::AbstractVecOrMat{T}, ids) where {T <: Union{Int32, Float64}}
    name == "mass" && throw(ArgumentError("scattering/gathering mass is currently not supported! Use `extract_atom()` instead."))

    dtype::Int = T === Float64 # 1 for Float64, 0 for Int32
    natoms = get_natoms(lmp)
    ndata::Int = isnothing(ids) ? natoms : length(ids)

    (type, count) = _get_type_and_count(lmp, name, ndims(data) == 2)

    if type in (API.LAMMPS_DOUBLE, API.LAMMPS_DOUBLE_2D)
        T !== Float64 && throw(ArgumentError("Expected a matrix with eltype `Float64` got eltype `Int32` instead."))
    elseif type in (API.LAMMPS_INT, API.LAMMPS_INT_2D)
        T !== Int32 && throw(ArgumentError("Expected a matrix with eltype `Int32` got eltype `Float64` instead."))
    else
        @assert false # this shouldn't be possible, I think ...
    end

    expected_size = count == 0 ? (ndata, ) : (count, ndata)
    size(data) == expected_size || throw(ArgumentError("expected array with size $expected_size got array of size $(size(data)) instead."))

    !_array_stride_valid(data) && throw(ArgumentError("data must be contiguous in memory (i.e., interpretable as a 1D array)"))
    
    if isnothing(ids)
        return lmp, name, dtype, max(count, 1), data
    else
        @assert all(1 <= id <= natoms for id in ids)
        return lmp, name, dtype, max(count, 1), ndata, ids, data
    end
end

"""
    gather(lmp::LMP, name::String, lmp_type::_LMP_DATATYPE [, ids::Array{Int32}])

Gather the named per-atom, per-atom fix, per-atom compute, or fix property/atom-based entities from all processes.
By default (when `ids=nothing`), this method collects data from all atoms in consecutive order according to their IDs.
The optional parameter `ids` determines for which subset of atoms the requested data will be gathered. The returned data will then be ordered according to `ids`

Compute entities have the prefix `c_`, fix entities use the prefix `f_`, and per-atom entites have no prefix.

| valid values for `lmp_type`: | resulting return type:   |
| :--------------------------- | :----------------------- |
| `LAMMPS_INT`                 | `Vector{Int32}`          |
| `LAMMPS_INT_2D`              | `Matrix{Int32}`          |
| `LAMMPS_DOUBLE`              | `Vector{Float64}`        |
| `LAMMPS_DOUBLE_2D`           | `Matrix{Float64}`        |


!!! info "ids"
    The optional parameter `ids` only works, if there is a map defined. For example by doing:
    `command(lmp, "atom_modify map yes")`

!!! note "image"
    for the per-atom property "image" either `LAMMPS_INT` or `LAMMPS_INT_2D` can be provided as the `lmp_type`,
    returning the encoded or decoded image flags, respectively.
"""
function gather(lmp::LMP, name::String, lmp_type::_LMP_DATATYPE, ids::Union{Nothing, Array{Int32}}=nothing)
    ndata::Int = isnothing(ids) ? get_natoms(lmp) : length(ids)
    
    (type, count) = _get_type_and_count(lmp, name, lmp_type === LAMMPS_INT_2D)
    count = max(1, count)

    expect = API._LMP_DATATYPE_CONST(type)
    receive = get_enum(lmp_type)
    expect != receive && throw(ArgumentError("TypeMismatch: Expected $expect got $receive instead!"))

    if lmp_type === LAMMPS_DOUBLE 
        data = Vector{Float64}(undef, ndata)
    elseif lmp_type === LAMMPS_INT
        data = Vector{Int32}(undef, ndata)
    elseif lmp_type === LAMMPS_DOUBLE_2D
        data = Matrix{Float64}(undef, count, ndata)
    elseif lmp_type === LAMMPS_INT_2D
        data = Matrix{Int32}(undef, count, ndata)
    else
        throw(ArgumentError("type $lmp_type is not supported for gather/scatter operations"))
    end

    gather!(lmp, name, data, ids)
end

"""
    gather!(lmp::LMP, name::String, data::AbstractVecOrMat{T} [, ids::Array{Int32}]) where {T <: Union{Int32, Float64}}

Gather the named per-atom, per-atom fix, per-atom compute, or fix property/atom-based entities from all processes and store the result in data.
By default (when `ids=nothing`), this method collects data from all atoms in consecutive order according to their IDs.
The optional parameter `ids` determines for which subset of atoms the requested data will be gathered. The returned data will then be ordered according to `ids`

Compute entities have the prefix `c_`, fix entities use the prefix `f_`, and per-atom entites have no prefix.

!!! info "ids"
    The optional parameter `ids` only works, if there is a map defined. For example by doing:
    `command(lmp, "atom_modify map yes")`

!!! note "image"
    for the per-atom property "image" either a `Vector{Int32}` or `Matrix{Int32}` can be used for the data array,
    representing the encoded or decoded image flags, respectively.
"""
function gather!(lmp::LMP, name::String, data::AbstractVecOrMat{T}, ids::Union{Nothing, Array{Int32}}=nothing) where {T <: Union{Int32, Float64}}
    param = _check_array(lmp, name, data, ids)
    isnothing(ids) ?
        API.lammps_gather(param...) :
        API.lammps_gather_subset(param...)
    check(lmp)
    return data
end

"""
    scatter!(lmp::LMP, name::String, data::AbstractVecOrMat{T} [, ids::Array{Int32}]) where T<:Union{Int32, Float64}

Scatter the named per-atom, per-atom fix, per-atom compute, or fix property/atom-based entity in data to all processes.
By default (when `ids=nothing`), this method scatters data to all atoms in consecutive order according to their IDs.
The optional parameter `ids` determines to which subset of atoms the data will be scattered.

Compute entities have the prefix `c_`, fix entities use the prefix `f_`, and per-atom entites have no prefix.

!!! info "ids"
    The optional parameter `ids` only works, if there is a map defined. For example by doing:
    `command(lmp, "atom_modify map yes")`

!!! note "image"
    for the per-atom property "image" either a `Vector{Int32}` or `Matrix{Int32}` can be used for the data array,
    representing the encoded or decoded image flags, respectively.
"""
function scatter!(lmp::LMP, name::String, data::AbstractVecOrMat{T}, ids::Union{Nothing, Array{Int32}}=nothing) where T<:Union{Int32, Float64}
    param = _check_array(lmp, name, data, ids)
    isnothing(ids) ?
        API.lammps_scatter(param...) :
        API.lammps_scatter_subset(param...)
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
    if v !== nothing && size(x) != size(v)
        throw(ArgumentError("x and v must be the same size"))
    end
    if image !== nothing && numAtoms != length(image)
        throw(ArgumentError("image must have the same length as the number of atoms"))
    end

    v = v === nothing ? C_NULL : v
    image = image === nothing ? C_NULL : image

    API.lammps_create_atoms(lmp, numAtoms, id, types, x, v, image, bexpand ? 1 : 0)
    check(lmp)
end

function gather(::LMP, ::String, ::Type, ::Union{Nothing, Array{Int32}}=nothing)
    throw("`gather(::LMP, ::String, ::Type [,ids])` is deprecated! " *
    "use `gather(::LMP, ::String, ::_LMP_TYPE [,ids])` instead!")
end