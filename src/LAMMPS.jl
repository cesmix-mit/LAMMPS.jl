module LAMMPS
import MPI
include("api.jl")
import .API: _LMP_STYLE_CONST, LMP_STYLE_GLOBAL, LMP_STYLE_ATOM, LMP_STYLE_LOCAL

export LMP, command, get_natoms, extract_atom, extract_compute, extract_global,
       extract_setting, gather, scatter!, group_to_atom_ids, get_category_ids,
       extract_variable,

       LAMMPS_NONE,
       LAMMPS_INT,
       LAMMPS_INT_2D,
       LAMMPS_DOUBLE,
       LAMMPS_DOUBLE_2D,
       LAMMPS_INT64,
       LAMMPS_INT64_2D,
       LAMMPS_STRING,

       TYPE_SCALAR,
       TYPE_VECTOR,
       TYPE_ARRAY,
       SIZE_VECTOR,
       SIZE_ROWS,
       SIZE_COLS,

       VAR_EQUAL,
       VAR_ATOM,
       VAR_VECTOR,
       VAR_STRING,

       LMP_STYLE_GLOBAL,
       LMP_STYLE_ATOM,
       LMP_STYLE_LOCAL

using Preferences

abstract type TypeEnum{N} end
get_enum(::TypeEnum{N}) where N = N

struct _LMP_DATATYPE{N} <: TypeEnum{N} end

const LAMMPS_NONE = _LMP_DATATYPE{API.LAMMPS_NONE}()
const LAMMPS_INT = _LMP_DATATYPE{API.LAMMPS_INT}()
const LAMMPS_INT_2D = _LMP_DATATYPE{API.LAMMPS_INT_2D}()
const LAMMPS_DOUBLE = _LMP_DATATYPE{API.LAMMPS_DOUBLE}()
const LAMMPS_DOUBLE_2D = _LMP_DATATYPE{API.LAMMPS_DOUBLE_2D}()
const LAMMPS_INT64 = _LMP_DATATYPE{API.LAMMPS_INT64}()
const LAMMPS_INT64_2D = _LMP_DATATYPE{API.LAMMPS_INT64_2D}()
const LAMMPS_STRING = _LMP_DATATYPE{API.LAMMPS_STRING}()

struct _LMP_TYPE{N} <: TypeEnum{N} end

const TYPE_SCALAR = _LMP_TYPE{API.LMP_TYPE_SCALAR}()
const TYPE_VECTOR = _LMP_TYPE{API.LMP_TYPE_VECTOR}()
const TYPE_ARRAY = _LMP_TYPE{API.LMP_TYPE_ARRAY}()
const SIZE_VECTOR = _LMP_TYPE{API.LMP_SIZE_VECTOR}()
const SIZE_ROWS = _LMP_TYPE{API.LMP_SIZE_ROWS}()
const SIZE_COLS = _LMP_TYPE{API.LMP_SIZE_COLS}()

struct LMP_VARIABLE{N} <: TypeEnum{N} end

const VAR_EQUAL = LMP_VARIABLE{API.LMP_VAR_EQUAL}()
const VAR_ATOM = LMP_VARIABLE{API.LMP_VAR_ATOM}()
const VAR_VECTOR = LMP_VARIABLE{API.LMP_VAR_VECTOR}()
const VAR_STRING = LMP_VARIABLE{API.LMP_VAR_STRING}()

"""
    locate()

Locate the LAMMPS library currently being used, by LAMMPS.jl
"""
locate() = API.LAMMPS_jll.get_liblammps_path()

"""
    set_library!(path)

Change the library path used by LAMMPS.jl for `liblammps.so` to `path`.

!!! note
    You will need to restart Julia to use the new library.

!!! warning
    Due to a bug in Julia (until 1.6.5 and 1.7.1), setting preferences in transitive dependencies
    is broken (https://github.com/JuliaPackaging/Preferences.jl/issues/24). To fix this either update
    your version of Julia, or add LAMMPS_jll as a direct dependency to your project.
"""
function set_library!(path)
    if !ispath(path)
        error("LAMMPS library path $path not found")
    end
    set_preferences!(
        API.LAMMPS_jll,
        "liblammps_path" => realpath(path);
        force=true,
    )
    @warn "LAMMPS library path changed, you will need to restart Julia for the change to take effect" path

    if VERSION <= v"1.6.5" || VERSION == v"1.7.0"
        @warn """
        Due to a bug in Julia (until 1.6.5 and 1.7.1), setting preferences in transitive dependencies
        is broken (https://github.com/JuliaPackaging/Preferences.jl/issues/24). To fix this either update
        your version of Julia, or add LAMMPS_jll as a direct dependency to your project.
        """
    end
end

"""
    LMP(args::Vector{String}=String[], comm::Union{Nothing, MPI.Comm}=nothing)

Create a new LAMMPS instance while passing in a list of strings as if they were command-line arguments for the LAMMPS executable.

For a full ist of Command-line options see: https://docs.lammps.org/Run_options.html
"""
mutable struct LMP
    @atomic handle::Ptr{Cvoid}

    function LMP(args::Vector{String}=String[], comm::Union{Nothing, MPI.Comm}=nothing)
        if !isempty(args)
            args = copy(args)
            pushfirst!(args, "lammps")
        end

        GC.@preserve args begin
            if comm !== nothing
                if !MPI.Initialized()
                    error("MPI has not been initialized. Make sure to first call `MPI.Init()`")
                end
                handle = API.lammps_open(length(args), args, comm, C_NULL)
            else
                handle = API.lammps_open_no_mpi(length(args), args, C_NULL)
            end
        end

        this = new(handle)
        finalizer(close!, this)
        return this
    end
end
Base.unsafe_convert(::Type{Ptr{Cvoid}}, lmp::LMP) = lmp.handle

"""
    close!(lmp::LMP)

Shutdown a LAMMPS instance.
"""
function close!(lmp::LMP)
    handle = @atomicswap lmp.handle = C_NULL
    if handle !== C_NULL 
       API.lammps_close(handle)
    end
end

"""
    LMP(f::Function, args=String[], comm=nothing)

Create a new LAMMPS instance and call `f` on that instance while returning the result from `f`.
"""
function LMP(f::Function, args=String[], comm=nothing)
    lmp = LMP(args, comm)
    return f(lmp)
    # `close!` is registered as a finalizer for LMP, no need to close it here.
end

function version(lmp::LMP)
    API.lammps_version(lmp)
end

function check(lmp::LMP)
    err = API.lammps_has_error(lmp)
    if err != 0
        # TODO: Check err == 1 or err == 2 (MPI)
        buf = zeros(UInt8, 100)
        API.lammps_get_last_error_message(lmp, buf, length(buf))
        error(rstrip(String(buf), '\0'))
    end
end


"""
    command(lmp::LMP, cmd::Union{String, Array{String}})

Process LAMMPS input commands from a String or from an Array of Strings.

For a full list of commands see: https://docs.lammps.org/commands_list.html

This function processes a multi-line string similar to a block of commands from a file.
The string may have multiple lines (separated by newline characters) and also single commands may
be distributed over multiple lines with continuation characters (’&’).
Those lines are combined by removing the ‘&’ and the following newline character.
After this processing the string is handed to LAMMPS for parsing and executing.

Arrays of Strings get concatenated into a single String inserting newline characters as needed.

!!! compat "LAMMPS.jl 0.4.1"
    Multiline string support `\"""` and support for array of strings was added.
    Prior versions of LAMMPS.jl ignore newline characters.

# Examples

```
LMP(["-screen", "none"]) do lmp
    command(lmp, \"""
        atom_modify map yes
        region cell block 0 2 0 2 0 2
        create_box 1 cell
        lattice sc 1
        create_atoms 1 region cell
        mass 1 1

        group a id 1 2 3 5 8
        group even id 2 4 6 8
        group odd id 1 3 5 7
    \""")
end
```
"""
function command(lmp::LMP, cmd::Union{String, Array{String}})
    if cmd isa String
        API.lammps_commands_string(lmp, cmd)
    else
        API.lammps_commands_list(lmp, length(cmd), cmd)
    end

    check(lmp)
end

"""
    get_natoms(lmp::LMP)::Int64

Get the total number of atoms in the LAMMPS instance.

Will be precise up to 53-bit signed integer due to the
underlying `lammps_get_natoms` returning a Float64.
"""
function get_natoms(lmp::LMP)
    Int64(API.lammps_get_natoms(lmp))
end

function _lammps_string(ptr::Ptr)
    ptr == C_NULL && error("Wrapping NULL-pointer!")
    return Base.unsafe_string(ptr)
end

function _lammps_wrap(ptr::Ptr{<:Real}, shape::Integer, copy=true)
    ptr == C_NULL && error("Wrapping NULL-pointer!")

    result = Base.unsafe_wrap(Array, ptr, shape, own=false)

    return copy ? Base.copy(result) : result
end

function _lammps_wrap(ptr::Ptr{<:Ptr{T}}, shape::NTuple{2}, copy=true) where T
    ptr == C_NULL && error("Wrapping NULL-pointer!")

    (count, ndata) = shape

    ndata == 0 && return Matrix{T}(undef, count, ndata) # There is no data that can be wrapped

    pointers = Base.unsafe_wrap(Array, ptr, ndata)

    # This assert verifies that all the pointers are evenly spaced according to count.
    # While it seems like this is allways the case, it's not explicitly stated in the
    # API documentation. It also helps to verify, that the count is correct.
    @assert all(diff(pointers) .== count*sizeof(T))

    # It the pointers are evenly spaced, we can simply use the first pointer to wrap our matrix.
    result = Base.unsafe_wrap(Array, pointers[1], shape, own=false)

    return copy ? Base.copy(result) : result
end

function _lammps_reinterpret(T::_LMP_DATATYPE, ptr::Ptr)
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

A full list of settings can be found here: <https://docs.lammps.org/Library_properties.html>

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
    return API.lammps_extract_setting(lmp, name)
end

"""
    extract_global(lmp::LMP, name::String, lmp_type::_LMP_DATATYPE; copy::Bool=true)

Extract a global property from a LAMMPS instance.

| valid values for `lmp_type`: | resulting return type: |
| :--------------------------- | :--------------------- |
| `LAMMPS_INT`                 | `Vector{Int32}`        |
| `LAMMPS_INT_2D`              | `Matrix{Int32}`        |
| `LAMMPS_DOUBLE`              | `Vector{Float64}`      |
| `LAMMPS_DOUBLE_2D`           | `Matrix{Float64}`      |
| `LAMMPS_INT64`               | `Vector{Int64}`        |
| `LAMMPS_INT64_2D`            | `Matrix{Int64}`        |
| `LAMMPS_STRING`              | `String`               |

the kwarg `copy`, which defaults to true, determies wheter a copy of the underlying data is made.
the pointer to the underlying data is generally persistent, unless a clear command is issued.
However it's still recommended to only disable this, if you wish to modify the internal state of the LAMMPS instance.

Scalar values get returned as a vector with a single element. This way it's possible to
modify the internal state of the LAMMPS instance even if the data is scalar.

A full list of global variables can be found here: <https://docs.lammps.org/Library_properties.html>
"""
function extract_global(lmp::LMP, name::String, lmp_type::_LMP_DATATYPE; copy::Bool=true)
    void_ptr = API.lammps_extract_global(lmp, name)
    void_ptr == C_NULL && error("Unknown global variable $name")

    expect = extract_global_datatype(lmp, name)
    recieve = get_enum(lmp_type)
    expect != recieve && error("TypeMismatch: Expected $expect got $recieve instead!")

    ptr = _lammps_reinterpret(lmp_type, void_ptr)

    lmp_type == LAMMPS_STRING && return _lammps_string(ptr)

    if name in ("boxlo", "boxhi", "sublo", "subhi", "sublo_lambda", "subhi_lambda", "periodicity")
        length = 3
    elseif name in ("special_lj", "special_coul")
        length = 4
    else
        length = 1
    end

    return _lammps_wrap(ptr, length, copy)
end

function extract_global_datatype(lmp::LMP, name)
    return API._LMP_DATATYPE_CONST(API.lammps_extract_global_datatype(lmp, name))
end

"""
    extract_atom(lmp::LMP, name::String, lmp_type::_LMP_DATATYPE; copy=true)

Extract per-atom data from the lammps instance.

| valid values for `lmp_type`: | resulting return type: |
| :--------------------------- | :--------------------- |
| `LAMMPS_INT`                 | `Vector{Int32}`        |
| `LAMMPS_INT_2D`              | `Matrix{Int32}`        |
| `LAMMPS_DOUBLE`              | `Vector{Float64}`      |
| `LAMMPS_DOUBLE_2D`           | `Matrix{Float64}`      |
| `LAMMPS_INT64`               | `Vector{Int64}`        |
| `LAMMPS_INT64_2D`            | `Matrix{Int64}`        |

the kwarg `copy`, which defaults to true, determies wheter a copy of the underlying data is made.
As the pointer to the underlying data is not persistent, it's highly recommended to only disable this,
if you wish to modify the internal state of the LAMMPS instance.

A table with suported name keywords can be found here: <https://docs.lammps.org/Classes_atom.html#_CPPv4N9LAMMPS_NS4Atom7extractEPKc>
"""
function extract_atom(lmp::LMP, name::String, lmp_type::_LMP_DATATYPE; copy=true)
    void_ptr = API.lammps_extract_atom(lmp, name)
    void_ptr == C_NULL && error("Unknown per-atom variable $name")

    expect = extract_atom_datatype(lmp, name)
    recieve = get_enum(lmp_type)
    expect != recieve && error("TypeMismatch: Expected $expect got $recieve instead!")

    ptr = _lammps_reinterpret(lmp_type, void_ptr)

    if name == "mass"
        length = extract_global(lmp, "ntypes", LAMMPS_INT, copy=false)[]
        ptr += sizeof(eltype(ptr)) # Scarry pointer arithemtic; The first entry in the array is unused
        return _lammps_wrap(ptr, length, copy)
    end

    length = extract_setting(lmp, "nlocal")

    if _is_2D_datatype(lmp_type)
        # only Quaternions have 4 entries
        # length is a Int32 and lammps_wrap expects a NTuple, so it's
        # neccecary to use Int32 for count as well
        count = name == "quat" ? Int32(4) : Int32(3)
        return _lammps_wrap(ptr, (count, length), copy)
    end

    return _lammps_wrap(ptr, length, copy)
end

function extract_atom_datatype(lmp::LMP, name)
    return API._LMP_DATATYPE_CONST(API.lammps_extract_atom_datatype(lmp, name))
end

"""
    extract_compute(lmp::LMP, name::String, style::_LMP_STYLE_CONST, lmp_type::_LMP_TYPE; copy::Bool=true)

Extract data provided by a compute command identified by the compute-ID.
Computes may provide global, per-atom, or local data, and those may be a scalar, a vector or an array.
Since computes may provide multiple kinds of data, it is required to set style and type flags representing what specific data is desired.

| valid values for `style`: |
| :------------------------ |
| `LMP_STYLE_GLOBAL`        |
| `LMP_STYLE_ATOM`          |
| `LMP_STYLE_LOCAL`         |

| valid values for `lmp_type`: | resulting return type: |
| :--------------------------- | :--------------------- |
| `TYPE_SCALAR`                | `Vector{Float64}`      |
| `TYPE_VECTOR`                | `Vector{Float64}`      |
| `TYPE_ARRAY`                 | `Matrix{Float64}`      |
| `SIZE_VECTOR`                | `Vector{Int32}`        |
| `SIZE_COLS`                  | `Vector{Int32}`        |
| `SIZE_ROWS`                  | `Vector{Int32}`        |

the kwarg `copy`, which defaults to true, determies wheter a copy of the underlying data is made.
As the pointer to the underlying data is not persistent, it's highly recommended to only disable this,
if you wish to modify the internal state of the LAMMPS instance.

Scalar values get returned as a vector with a single element. This way it's possible to
modify the internal state of the LAMMPS instance even if the data is scalar.

# Examples

```julia
    LMP(["-screen", "none"]) do lmp
        extract_compute(lmp, "thermo_temp", LMP_STYLE_GLOBAL, TYPE_VECTOR)[2] = 2
        extract_compute(lmp, "thermo_temp", LMP_STYLE_GLOBAL, TYPE_VECTOR, copy=false)[3] = 3

        extract_compute(lmp, "thermo_temp", LMP_STYLE_GLOBAL, TYPE_SCALAR) |> println # [0.0]
        extract_compute(lmp, "thermo_temp", LMP_STYLE_GLOBAL, TYPE_VECTOR) |> println # [0.0, 0.0, 3.0, 0.0, 0.0, 0.0]
    end
```
"""
function extract_compute(lmp::LMP, name::String, style::_LMP_STYLE_CONST, lmp_type::_LMP_TYPE; copy::Bool=true)
    API.lammps_has_id(lmp, "compute", name) != 1 && error("Unknown compute $name")

    void_ptr = API.lammps_extract_compute(lmp, name, style, get_enum(lmp_type))
    void_ptr == C_NULL && error("Compute $name doesn't have data matching $style, $(get_enum(lmp_type))")

    # `lmp_type in (SIZE_COLS, SIZE_ROWS, SIZE_VECTOR)` causes type instability for some reason
    if lmp_type == SIZE_COLS || lmp_type == SIZE_ROWS || lmp_type == SIZE_VECTOR
        ptr = _lammps_reinterpret(LAMMPS_INT, void_ptr)
        return _lammps_wrap(ptr, 1, copy)
    end

    if lmp_type == TYPE_SCALAR
        ptr = _lammps_reinterpret(LAMMPS_DOUBLE, void_ptr)
        return _lammps_wrap(ptr, 1, copy)
    end

    if lmp_type == TYPE_VECTOR
        ndata = (style == LMP_STYLE_ATOM) ?
            extract_setting(lmp, "nlocal") :
            extract_compute(lmp, name, style, SIZE_VECTOR, copy=false)[]

        ptr = _lammps_reinterpret(LAMMPS_DOUBLE, void_ptr)
        return _lammps_wrap(ptr, ndata, copy)
    end

    ndata = (style == LMP_STYLE_ATOM) ?
        extract_setting(lmp, "nlocal") :
        extract_compute(lmp, name, style, SIZE_ROWS, copy=false)[]

    count = extract_compute(lmp, name, style, SIZE_COLS, copy=false)[]
    ptr = _lammps_reinterpret(LAMMPS_DOUBLE_2D, void_ptr)

    return _lammps_wrap(ptr, (count, ndata), copy)
end

"""
    extract_variable(lmp::LMP, name::String, lmp_variable::LMP_VARIABLE, group::Union{String, Nothing}=nothing; copy::Bool=true)

Extracts the data from a LAMMPS variable. When the variable is either an `equal`-style compatible variable,
a `vector`-style variable, or an `atom`-style variable, the variable is evaluated and the corresponding value(s) returned.
Variables of style `internal` are compatible with `equal`-style variables, if they return a numeric value.
For other variable styles, their string value is returned.

| valid values for `lmp_variable`: | return type       |
| :------------------------------- | :---------------  |
| `VAR_ATOM`                       | `Vector{Float64}` |
| `VAR_EQUAL`                      | `Float64`         |
| `VAR_STRING`                     | `String`          |
| `VAR_VECTOR`                     | `Vector{Float64}` |

the kwarg `copy`, which defaults to true, determies wheter a copy of the underlying data is made.
`copy` is only aplicable for `VAR_VECTOR`. For all other variable types, a copy will be made regardless.

the kwarg `group` determines for which atoms the variable will be extracted. It's only aplicable for
`VAR_ATOM` and will cause an error if used for other variable types. The entires for all atoms not in the group
will be zeroed out. By default, all atoms will be extracted.
"""
function extract_variable(lmp::LMP, name::String, lmp_variable::LMP_VARIABLE, group::Union{String, Nothing}=nothing; copy::Bool=true)
    lmp_variable != VAR_ATOM && !isnothing(group) && error("the group parameter is only supported for per atom variables!")

    if isnothing(group)
        group = C_NULL
    end

    void_ptr = API.lammps_extract_variable(lmp, name, group)
    void_ptr == C_NULL && error("Unknown variable $name")

    expect = extract_variable_datatype(lmp, name)
    recieve = get_enum(lmp_variable)
    if expect != recieve
        if lmp_variable == VAR_EQUAL || lmp_variable == VAR_ATOM
            API.lammps_Free(void_ptr)
        end

        error("TypeMismatch: Expected $expect got $recieve instead!")
    end

    if lmp_variable == VAR_EQUAL
        ptr = _lammps_reinterpret(LAMMPS_DOUBLE, void_ptr)
        result = unsafe_load(ptr)
        API.lammps_free(ptr)
        return result
    end

    if lmp_variable == VAR_VECTOR
        ndata_ptr = _lammps_reinterpret(LAMMPS_INT, API.lammps_extract_variable(lmp, name, "GET_VECTOR_SIZE"))
        ndata = unsafe_load(ndata_ptr)
        API.lammps_free(ndata_ptr)

        ptr = _lammps_reinterpret(LAMMPS_DOUBLE, void_ptr)
        return _lammps_wrap(ptr, ndata, copy)
    end

    if lmp_variable == VAR_ATOM
        ndata = extract_setting(lmp, "nlocal")

        ptr = _lammps_reinterpret(LAMMPS_DOUBLE, void_ptr)
        result = _lammps_wrap(ptr, ndata, true)
        API.lammps_free(ptr)
        return result
    end

    ptr = _lammps_reinterpret(LAMMPS_STRING, void_ptr)
    return _lammps_string(ptr)
end

function extract_variable_datatype(lmp::LMP, name)
    return API._LMP_VAR_CONST(API.lammps_extract_variable_datatype(lmp, name))
end


@deprecate gather_atoms(lmp::LMP, name, T, count) gather(lmp, name, T)


"""
    gather(lmp::LMP, name::String, T::Union{Type{Int32}, Type{Float64}}, ids::Union{Nothing, Array{Int32}}=nothing)

Gather the named per-atom, per-atom fix, per-atom compute, or fix property/atom-based entities from all processes.
By default (when `ids=nothing`), this method collects data from all atoms in consecutive order according to their IDs.
The optional parameter `ids` determines for which subset of atoms the requested data will be gathered. The returned data will then be ordered according to `ids`

Compute entities have the prefix `c_`, fix entities use the prefix `f_`, and per-atom entites have no prefix.

The returned Array is decoupled from the internal state of the LAMMPS instance.

!!! warning "Type Verification"
    Due to how the underlying C-API works, it's not possible to verify the element data-type of fix or compute style data.
    Supplying the wrong data-type will not throw an error but will result in nonsensical output

!!! warning "ids"
    The optional parameter `ids` only works, if there is a map defined. For example by doing:
    `command(lmp, "atom_modify map yes")`
    However, LAMMPS only issues a warning if that's the case, which unfortuately cannot be detected through the underlying API.
    Starting form LAMMPS version `17 Apr 2024` this should no longer be an issue, as LAMMPS then throws an error instead of a warning.
"""
function gather(lmp::LMP, name::String, T::Union{Type{Int32}, Type{Float64}}, ids::Union{Nothing, Array{Int32}}=nothing)
    name == "mass" && error("scattering/gathering mass is currently not supported! Use `extract_atom()` instead.")

    count = _get_count(lmp, name)
    _T = _get_T(lmp, name)

    @assert ismissing(_T) || _T == T "Expected data type $_T got $T instead."

    dtype = (T === Float64)
    natoms = get_natoms(lmp)
    ndata = isnothing(ids) ? natoms : length(ids)
    data = Matrix{T}(undef, (count, ndata))

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
    scatter!(lmp::LMP, name::String, data::VecOrMat{T}, ids::Union{Nothing, Array{Int32}}=nothing) where T<:Union{Int32, Float64}

Scatter the named per-atom, per-atom fix, per-atom compute, or fix property/atom-based entity in data to all processes.
By default (when `ids=nothing`), this method scatters data to all atoms in consecutive order according to their IDs.
The optional parameter `ids` determines to which subset of atoms the data will be scattered.

Compute entities have the prefix `c_`, fix entities use the prefix `f_`, and per-atom entites have no prefix.

!!! warning "Type Verification"
    Due to how the underlying C-API works, it's not possible to verify the element data-type of fix or compute style data.
    Supplying the wrong data-type will not throw an error but will result in nonsensical date being supplied to the LAMMPS instance.

!!! warning "ids"
    The optional parameter `ids` only works, if there is a map defined. For example by doing:
    `command(lmp, "atom_modify map yes")`
    However, LAMMPS only issues a warning if that's the case, which unfortuately cannot be detected through the underlying API.
    Starting form LAMMPS version `17 Apr 2024` this should no longer be an issue, as LAMMPS then throws an error instead of a warning.
"""
function scatter!(lmp::LMP, name::String, data::VecOrMat{T}, ids::Union{Nothing, Array{Int32}}=nothing) where T<:Union{Int32, Float64}
    name == "mass" && error("scattering/gathering mass is currently not supported! Use `extract_atom()` instead.")

    count = _get_count(lmp, name)
    _T = _get_T(lmp, name)

    @assert ismissing(_T) || _T == T "Expected data type $_T got $T instead."

    dtype = (T === Float64)
    natoms = get_natoms(lmp)
    ndata = isnothing(ids) ? natoms : length(ids)

    if data isa Vector
        @assert count == 1
        @assert ndata == lenght(data)
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
        return count == 0 ? 1 : count
    elseif name in ("mass", "id", "type", "mask", "image", "molecule", "q", "radius", "rmass", "ellipsoid", "line", "tri", "body", "temperature", "heatflow")
        return 1
    elseif name in ("x", "v", "f", "mu", "omega", "angmom", "torque")
        return 3
    elseif name == "quat"
        return 4
    else
        error("Unknown per atom property $name")
    end
end

function _get_T(lmp::LMP, name::String)
    if startswith(name, r"[f,c]_")
        return missing # As far as I know, it's not possible to determine the datatype of computes or fixes at runtime
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

"""
    group_to_atom_ids(lmp::LMP, group::String)

Find the IDs of the Atoms in the group.
"""
function group_to_atom_ids(lmp::LMP, group::String)
    # Pad with '\0' to avoid confusion with groups names that are truncated versions of name
    # For example 'all' could be confused with 'a'
    name_padded = codeunits(group * '\0')
    buffer_size = length(name_padded)
    buffer = zeros(UInt8, buffer_size)

    ngroups = API.lammps_id_count(lmp, "group")
    
    for idx in 0:ngroups-1
        API.lammps_id_name(lmp, "group", idx, buffer, buffer_size)
        buffer != name_padded && continue

        mask = gather(lmp, "mask", Int32)[:] .& (1 << idx) .!= 0
        all_ids = UnitRange{Int32}(1, get_natoms(lmp))

        return all_ids[mask]
    end

    error("Cannot find group $group")
end


"""
    get_category_ids(lmp::LMP, category::String, buffer_size::Integer=50)

Look up the names of entities within a certain category.

Valid categories are: compute, dump, fix, group, molecule, region, and variable.
names longer than `buffer_size` will be truncated to fit inside the buffer.
"""
function get_category_ids(lmp::LMP, category::String, buffer_size::Integer=50)
    _check_valid_category(category)

    count = API.lammps_id_count(lmp, category)
    check(lmp)

    res = Vector{String}(undef, count)

    for i in 1:count
        buffer = zeros(UInt8, buffer_size)
        API.lammps_id_name(lmp, category, i-1, buffer, buffer_size)
        res[i] = rstrip(String(buffer), '\0')
    end

    return res
end

_check_valid_category(category::String) = category in ("compute", "dump", "fix", "group", "molecule", "region", "variable") || error("$category is not a valid category name!")

end # module
