function _string(ptr::Ptr)
    ptr == C_NULL && error("Wrapping NULL-pointer!")
    return Base.unsafe_string(ptr)
end

function _extract(ptr::Ptr{<:Real})
    ptr == C_NULL && error("Wrapping NULL-pointer!")
    return UnsafeArray(ptr, ())
end

function _extract(ptr::Ptr{<:Real}, shape::Integer)
    ptr == C_NULL && error("Wrapping NULL-pointer!")
    return UnsafeArray(ptr, Tuple{Int}(shape))
end

function _extract(ptr::Ptr{<:Ptr{T}}, shape::NTuple{2}) where T
    ptr == C_NULL && error("Wrapping NULL-pointer!")
    first_pointer::Ptr{T} = prod(shape) == 0 ? C_NULL : unsafe_load(ptr)
    return UnsafeArray(first_pointer, Int.(shape))
end

function _reinterpret(T::_LMP_DATATYPE, ptr::Ptr)
    T === LAMMPS_INT && return Base.reinterpret(Ptr{Int32}, ptr)
    T === LAMMPS_INT_2D && return Base.reinterpret(Ptr{Ptr{Int32}}, ptr)
    T === LAMMPS_DOUBLE && return Base.reinterpret(Ptr{Float64}, ptr)
    T === LAMMPS_DOUBLE_2D && return Base.reinterpret(Ptr{Ptr{Float64}}, ptr)
    T === LAMMPS_INT64 && return Base.reinterpret(Ptr{Int64}, ptr)
    T === LAMMPS_INT64_2D && return Base.reinterpret(Ptr{Ptr{Int64}}, ptr)
    T === LAMMPS_STRING && return Base.reinterpret(Ptr{UInt8}, ptr)
end

_is_2D_datatype(lmp_dtype::_LMP_DATATYPE) = lmp_dtype in (LAMMPS_INT_2D, LAMMPS_DOUBLE_2D, LAMMPS_INT64_2D)

"""
    extract_setting(lmp::LMP, name::String)::Int32

Query LAMMPS about global settings.

A full list of settings can be found in the [lammps documentation](https://docs.lammps.org/Library_properties.html).

# Examples
```julia
    LMP(["-screen", "none"]) do lmp
        command(lmp, \"""
            region cell block 0 3 0 3 0 3
            create_box 1 cell
            lattice sc 1
            create_atoms 1 region cell
        \""")

        extract_setting(lmp, "dimension") |> println # 3
        extract_setting(lmp, "nlocal") |> println # 27
    end
```
"""
function extract_setting(lmp::LMP, name::String)::Int32
    val = API.lammps_extract_setting(lmp, name)
    val == -1 && error("Could not find setting $name")
    return val
end

"""
    extract_global(lmp::LMP, name::String, lmp_type::_LMP_DATATYPE)

Extract a global property from a LAMMPS instance.

| valid values for `lmp_type`: | resulting return type:    |
| :--------------------------- | :------------------------ |
| `LAMMPS_INT`                 | `UnsafeArray{Int32, 1}`   |
| `LAMMPS_DOUBLE`              | `UnsafeArray{Float64, 1}` |
| `LAMMPS_INT64`               | `UnsafeArray{Int64, 1}`   |
| `LAMMPS_STRING`              | `String`                  |

Scalar values get returned as a vector with a single element. This way it's possible to
modify the internal state of the LAMMPS instance even if the data is scalar.

!!! info
    Closing the LAMMPS instance or issuing a clear command after calling this method
    will result in the returned data becoming invalid. To prevent this, copy the returned data.

!!! warning
    Modifying the data through `extract_global` may lead to inconsistent internal data and thus may cause failures or crashes or bogus simulations.
    In general it is thus usually better to use a LAMMPS input command that sets or changes these parameters.
    Those will take care of all side effects and necessary updates of settings derived from such settings.

A full list of global variables can be found in the [lammps documentation](https://docs.lammps.org/Library_properties.html).
"""
function extract_global(lmp::LMP, name::String, lmp_type::_LMP_DATATYPE)
    void_ptr = API.lammps_extract_global(lmp, name)
    void_ptr == C_NULL && throw(KeyError("Unknown global variable $name"))

    expect = extract_global_datatype(lmp, name)
    receive = get_enum(lmp_type)
    expect != receive && error("TypeMismatch: Expected $expect got $receive instead!")

    ptr = _reinterpret(lmp_type, void_ptr)

    lmp_type == LAMMPS_STRING && return _string(ptr)

    if name in ("boxlo", "boxhi", "sublo", "subhi", "sublo_lambda", "subhi_lambda", "periodicity")
        length = 3
    elseif name in ("special_lj", "special_coul")
        length = 4
    else
        length = 1
    end

    return _extract(ptr, length)
end

function extract_global_datatype(lmp::LMP, name)
    return API._LMP_DATATYPE_CONST(API.lammps_extract_global_datatype(lmp, name))
end

struct LammpsBox
    boxlo::NTuple{3, Float64}
    boxhi::NTuple{3, Float64}
    xy::Float64
    yz::Float64
    xz::Float64
    pflags::NTuple{3, Int32}
    boxflag::Int32
end

"""
    extract_box(lmp::LMP)

Extract simulation box parameters.

Returns a `LammpsBox` containing the following fields:
 - `boxlo::NTuple{3, Float64}`
 - `boxhi::NTuple{3, Float64}`
 - `xy::Float64`
 - `yz::Float64`
 - `xz::Float64`
 - `pflags::NTuple{3, Int32}`
 - `boxflag::Int32`
"""
function extract_box(lmp::LMP)
    boxlo = Ref{NTuple{3, Float64}}()
    boxhi = Ref{NTuple{3, Float64}}()
    xy = Ref{Float64}()
    yz = Ref{Float64}()
    xz = Ref{Float64}()
    pflags = Ref{NTuple{3, Int32}}()
    boxflag = Ref{Int32}()

    @inline API.lammps_extract_box(lmp, boxlo, boxhi, xy, yz, xz, pflags, boxflag)
    check(lmp)
    return LammpsBox(boxlo[], boxhi[], xy[], yz[], xz[], pflags[], boxflag[])
end

"""
    reset_box(lmp::LMP, boxlo, boxhi, xy::Real = 0, yz::Real = 0, xz::Real = 0)

Reset simulation box parameters.
"""
function reset_box(lmp::LMP, boxlo, boxhi, xy::Real = 0, yz::Real = 0, xz::Real = 0)
    _boxlo = Ref(NTuple{3, Float64}(boxlo))
    _boxhi = Ref(NTuple{3, Float64}(boxhi))
    @inline API.lammps_reset_box(lmp, _boxlo, _boxhi, xy, yz, xz)
    check(lmp)
end

"""
    extract_atom(lmp::LMP, name::String, lmp_type::_LMP_DATATYPE; with_ghosts=false)

Extract per-atom data from the lammps instance.

| valid values for `lmp_type`: | resulting return type:   |
| :--------------------------- | :----------------------- |
| `LAMMPS_INT`                 | `UnsafeArray{Int32, 1}`  |
| `LAMMPS_INT_2D`              | `UnsafeArray{Int32, 2}`  |
| `LAMMPS_DOUBLE`              | `UnsafeArray{Float64, 1}`|
| `LAMMPS_DOUBLE_2D`           | `UnsafeArray{Float64, 2}`|
| `LAMMPS_INT64`               | `UnsafeArray{Int64, 1}`  |
| `LAMMPS_INT64_2D`            | `UnsafeArray{Int64, 2}`  |

!!! info
    The returned data may become invalid if a re-neighboring operation
    is triggered at any point after calling this method. If this has happened,
    trying to read from this data will likely cause julia to crash.
    To prevent this, copy the returned data

A table with suported name keywords can be found in the [lammps documentation](https://docs.lammps.org/Classes_atom.html#_CPPv4N9LAMMPS_NS4Atom7extractEPKc).

## Arguments
- `with_ghosts`: Determines wheter entries for ghost atoms are included. This is ignored for "mass", or when there is no ghost atom data available.
"""
function extract_atom(lmp::LMP, name::String, lmp_type::_LMP_DATATYPE; with_ghosts=false)
    void_ptr = API.lammps_extract_atom(lmp, name)
    void_ptr == C_NULL && throw(KeyError("Unknown per-atom variable $name"))

    expect = extract_atom_datatype(lmp, name)
    receive = get_enum(lmp_type)
    expect != receive && error("TypeMismatch: Expected $expect got $receive instead!")

    ptr = _reinterpret(lmp_type, void_ptr)

    if name == "mass"
        length = extract_global(lmp, "ntypes", LAMMPS_INT)[]
        ptr += sizeof(eltype(ptr)) # Scarry pointer arithemtic; The first entry in the array is unused
        return _extract(ptr, length)
    end

    length = if with_ghosts
        API.lammps_extract_atom_size(lmp, name, API.LMP_SIZE_ROWS)
    else
        extract_setting(lmp, "nlocal")
    end

    if _is_2D_datatype(lmp_type)
        count = API.lammps_extract_atom_size(lmp, name, API.LMP_SIZE_COLS)
        return _extract(ptr, (count, length))
    end

    return _extract(ptr, length)
end

function extract_atom_datatype(lmp::LMP, name)
    return API._LMP_DATATYPE_CONST(API.lammps_extract_atom_datatype(lmp, name))
end

"""
    extract_compute(lmp::LMP, name::String, style::_LMP_STYLE_CONST, lmp_type::_LMP_TYPE)

Extract data provided by a compute command identified by the compute-ID.
Computes may provide global, per-atom, or local data, and those may be a scalar, a vector or an array.
Since computes may provide multiple kinds of data, it is required to set style and type flags representing what specific data is desired.

| valid values for `style`: |
| :------------------------ |
| `STYLE_GLOBAL`            |
| `STYLE_ATOM`              |
| `STYLE_LOCAL`             |

| valid values for `lmp_type`: | resulting return type:   |
| :--------------------------- | :----------------------- |
| `TYPE_SCALAR`                | `UnsafeArray{Float64, 0}`|
| `TYPE_VECTOR`                | `UnsafeArray{Float64, 1}`|
| `TYPE_ARRAY`                 | `UnsafeArray{Float64, 2}`|

Scalar values get returned as arrays with a single element. This way it's possible to
modify the internal state of the LAMMPS instance even if the data is scalar.

!!! info
    The returned data may become invalid as soon as another LAMMPS command has been issued at any point after calling this method.
    If this has happened, trying to read from this data will likely cause julia to crash.
    To prevent this, copy the returned data.

# Examples

```julia
LMP(["-screen", "none"]) do lmp
    extract_compute(lmp, "thermo_temp", LMP_STYLE_GLOBAL, TYPE_SCALAR) |> println # [0.0]
end
```
"""
function extract_compute(lmp::LMP, name::String, style::_LMP_STYLE_CONST, lmp_type::_LMP_TYPE)
    void_ptr = API.lammps_extract_compute(lmp, name, style, get_enum(lmp_type))
    check(lmp)

    # `lmp_type in (SIZE_COLS, SIZE_ROWS, SIZE_VECTOR)` causes type instability for some reason
    if lmp_type == SIZE_COLS || lmp_type == SIZE_ROWS || lmp_type == SIZE_VECTOR
        ptr = _reinterpret(LAMMPS_INT, void_ptr)
        return _extract(ptr, 1)
    end

    if lmp_type == TYPE_SCALAR
        ptr = _reinterpret(LAMMPS_DOUBLE, void_ptr)
        return _extract(ptr, 1)
    end

    if lmp_type == TYPE_VECTOR
        ndata = (style == STYLE_ATOM) ?
            extract_setting(lmp, "nlocal") :
            extract_compute(lmp, name, style, SIZE_VECTOR)[]

        ptr = _reinterpret(LAMMPS_DOUBLE, void_ptr)
        return  _extract(ptr, ndata)
    end

    ndata = (style == STYLE_ATOM) ?
        extract_setting(lmp, "nlocal") :
        extract_compute(lmp, name, style, SIZE_ROWS)[]

    count = extract_compute(lmp, name, style, SIZE_COLS)[]
    ptr = _reinterpret(LAMMPS_DOUBLE_2D, void_ptr)

    return _extract(ptr, (count, ndata))
end

"""
    extract_variable(lmp::LMP, name::String, lmp_variable::LMP_VARIABLE, group::Union{String, Nothing}=nothing)

Extracts the data from a LAMMPS variable. When the variable is either an `equal`-style compatible variable,
a `vector`-style variable, or an `atom`-style variable, the variable is evaluated and the corresponding value(s) returned.
Variables of style `internal` are compatible with `equal`-style variables, if they return a numeric value.
For other variable styles, their string value is returned.

| valid values for `lmp_variable`: | return type              |
| :------------------------------- | :----------------------  |
| `VAR_ATOM`                       | `Vector{Float64}`(copy)  |
| `VAR_EQUAL`                      | `Float64`                |
| `VAR_STRING`                     | `String`                 |
| `VAR_VECTOR`                     | `UnsafeArray{Float64, 1}`|

the kwarg `group` determines for which atoms the variable will be extracted. It's only aplicable for
`VAR_ATOM` and will cause an error if used for other variable types. The entires for all atoms not in the group
will be zeroed out. By default, all atoms will be extracted.
"""
function extract_variable(lmp::LMP, name::String, lmp_variable::_LMP_VARIABLE, group::Union{String, Nothing}=nothing)
    lmp_variable != VAR_ATOM && !isnothing(group) && throw(ArgumentError("the group parameter is only supported for per atom variables!"))

    if isnothing(group)
        group = C_NULL
    end

    void_ptr = API.lammps_extract_variable(lmp, name, group)
    check(lmp)

    expect = extract_variable_datatype(lmp, name)
    receive = get_enum(lmp_variable)
    if expect != receive
        # the documentation instructs us to free the pointers for these styles specifically
        if expect in (API.LMP_VAR_ATOM, API.LMP_VAR_EQUAL)
            API.lammps_free(void_ptr)
        end

        error("TypeMismatch: Expected $expect got $receive instead!")
    end

    if lmp_variable == VAR_EQUAL
        ptr = _reinterpret(LAMMPS_DOUBLE, void_ptr)
        result = unsafe_load(ptr)
        API.lammps_free(ptr)
        return result
    end

    if lmp_variable == VAR_VECTOR
        # Calling lammps_extract_variable directly through the API instead of the higher level wrapper, as
        # "LMP_SIZE_VECTOR" is the only group name that won't be ignored for Vector Style Variables.
        # This isn't exposed to the high level API as it causes type instability for something that probably won't
        # ever be used outside of this implementation
        ndata_ptr = _reinterpret(LAMMPS_INT, API.lammps_extract_variable(lmp, name, "LMP_SIZE_VECTOR"))
        ndata = unsafe_load(ndata_ptr)
        API.lammps_free(ndata_ptr)

        ptr = _reinterpret(LAMMPS_DOUBLE, void_ptr)
        return _extract(ptr, ndata)
    end

    if lmp_variable == VAR_ATOM
        ndata = extract_setting(lmp, "nlocal")
        ptr = _reinterpret(LAMMPS_DOUBLE, void_ptr)
        result = copy(_extract(ptr, ndata))
        LAMMPS.API.lammps_free(ptr) # lammps expects us to take ownership of the data
        return result
    end

    ptr = _reinterpret(LAMMPS_STRING, void_ptr)
    return _string(ptr)
end

function extract_variable_datatype(lmp::LMP, name)
    res = API.lammps_extract_variable_datatype(lmp, name)
    check(lmp)
    return API._LMP_VAR_CONST(res)
end