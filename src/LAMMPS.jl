module LAMMPS
import MPI
include("api.jl")

export LMP, command, create_atoms, get_natoms, extract_atom, extract_compute, extract_global,
       extract_setting, gather, gather_bonds, gather_angles, gather_dihedrals, gather_impropers,
       scatter!, group_to_atom_ids, get_category_ids, extract_variable, LAMMPSError,

       # _LMP_TYPE
       TYPE_SCALAR,
       TYPE_VECTOR,
       TYPE_ARRAY,

       # _LMP_VARIABLE
       VAR_EQUAL,
       VAR_ATOM,
       VAR_VECTOR,
       VAR_STRING,

       # _LMP_STYLE_CONST
       STYLE_GLOBAL,
       STYLE_ATOM,
       STYLE_LOCAL,

       # LAMMPS to Julia types
       BIGINT,
       TAGINT,
       IMAGEINT

using Preferences

abstract type TypeEnum{N} end
get_enum(::TypeEnum{N}) where N = N

struct _LMP_TYPE{N} <: TypeEnum{N} end

const TYPE_SCALAR = _LMP_TYPE{API.LMP_TYPE_SCALAR}()
const TYPE_VECTOR = _LMP_TYPE{API.LMP_TYPE_VECTOR}()
const TYPE_ARRAY = _LMP_TYPE{API.LMP_TYPE_ARRAY}()
const SIZE_VECTOR = _LMP_TYPE{API.LMP_SIZE_VECTOR}()
const SIZE_ROWS = _LMP_TYPE{API.LMP_SIZE_ROWS}()
const SIZE_COLS = _LMP_TYPE{API.LMP_SIZE_COLS}()

struct _LMP_VARIABLE{N} <: TypeEnum{N} end

const VAR_EQUAL = _LMP_VARIABLE{API.LMP_VAR_EQUAL}()
const VAR_ATOM = _LMP_VARIABLE{API.LMP_VAR_ATOM}()
const VAR_VECTOR = _LMP_VARIABLE{API.LMP_VAR_VECTOR}()
const VAR_STRING = _LMP_VARIABLE{API.LMP_VAR_STRING}()

# these are not defined as TypeEnum as they don't carry type information
const _LMP_STYLE_CONST = API._LMP_STYLE_CONST

const STYLE_GLOBAL = API.LMP_STYLE_GLOBAL
const STYLE_ATOM = API.LMP_STYLE_ATOM
const STYLE_LOCAL = API.LMP_STYLE_LOCAL

const BIGINT = API.lammps_extract_setting(C_NULL, :bigint) == 4 ? Int32 : Int64
const TAGINT = API.lammps_extract_setting(C_NULL, :tagint) == 4 ? Int32 : Int64
const IMAGEINT = API.lammps_extract_setting(C_NULL, :imageint) == 4 ? Int32 : Int64

# don't use or alter this in any way except for `extract_atom_datatype` !!!
const LAMMPS_DUMMY = Ref{Ptr{Cvoid}}(0)

function __init__()
    BIGINT != (API.lammps_extract_setting(C_NULL, "bigint") == 4 ? Int32 : Int64) &&
        error("The size of the LAMMPS integer type BIGINT has changed! To fix this, you need to manually invalidate the LAMMPS.jl cache.")
    TAGINT != (API.lammps_extract_setting(C_NULL, "tagint") == 4 ? Int32 : Int64) &&
        error("The size of the LAMMPS integer type TAGINT has changed! To fix this, you need to manually invalidate the LAMMPS.jl cache.")
    IMAGEINT != (API.lammps_extract_setting(C_NULL, "tagint") == 4 ? Int32 : Int64) &&
        error("The size of the LAMMPS integer type IMAGEINT has changed! To fix this, you need to manually invalidate the LAMMPS.jl cache.")


    args = ["lammps", "-screen", "none", "-log", "none"]
    LAMMPS_DUMMY[] = API.lammps_open_no_mpi(length(args), args, C_NULL)
end


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

A full ist of command-line options can be found in the [lammps documentation](https://docs.lammps.org/Run_options.html).
"""
mutable struct LMP
    @atomic handle::Ptr{Cvoid}

    function LMP(args::Vector{String}=String[], comm::Union{Nothing, MPI.Comm}=nothing)
        args = copy(args)
        pushfirst!(args, "lammps")

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

        if API.lammps_has_error(handle) != 0
            buf = zeros(UInt8, 100)
            API.lammps_get_last_error_message(handle, buf, length(buf))
            msg = replace(rstrip(String(buf), '\0'), "ERROR: " => "")
            throw(LAMMPSError(msg))
        end

        this = new(handle)
        finalizer(close!, this)
        return this
    end
end

function Base.cconvert(::Type{Ptr{Cvoid}}, lmp::LMP)    
    lmp.handle == C_NULL && error("The LMP object doesn't point to a valid LAMMPS instance! "
            * "This is usually caused by calling `LAMMPS.close!` or through serialization and deserialization.")
    return lmp
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

struct LAMMPSError <: Exception
    msg::String
end

function LAMMPSError(lmp::LMP)
    buf = zeros(UInt8, 100)
    API.lammps_get_last_error_message(lmp, buf, length(buf))
    msg = replace(rstrip(String(buf), '\0'), "ERROR: " => "")
    LAMMPSError(msg)
end

function Base.showerror(io::IO, err::LAMMPSError)
    print(io, "LAMMPSError: ", err.msg)
end

function check(lmp::LMP)
    err = API.lammps_has_error(lmp)
    # TODO: Check err == 1 or err == 2 (MPI)
    if err != 0
        throw(LAMMPSError(lmp))
    end
end

"""
    command(lmp::LMP, cmd::Union{String, Array{String}})

Process LAMMPS input commands from a String or from an Array of Strings.

A full list of commands can be found in the [lammps documentation](https://docs.lammps.org/commands_list.html).

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

```julia
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
    create_atoms(
        lmp::LMP, x::Matrix{Float64}, id::Vector{Int32}, types::Vector{Int32};
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
    lmp::LMP, x::Matrix{Float64}, id::Vector{Int32}, types::Vector{Int32};
    v::Union{Nothing,Matrix{Float64}}=nothing,
    image::Union{Nothing,Vector{IMAGEINT}}=nothing,
    bexpand::Bool=false
)
    numAtoms = size(x, 2)
    if size(x, 1) != 3
        throw(ArgumentError("x must be a n by 3 matrix, where n is the number of atoms"))
    end
    if numAtoms != length(id)
        throw(ArgumentError("id must have the same length as the number of atoms"))
    end
    if numAtoms != length(types)
        throw(ArgumentError("types must have the same length as the number of atoms"))
    end
    if !isnothing(v) && size(x) != size(v)
        throw(ArgumentError("x and v must be the same size"))
    end
    if !isnothing(image) && numAtoms != length(image)
        throw(ArgumentError("image must have the same length as the number of atoms"))
    end

    v = isnothing(v) ? C_NULL : v
    image = isnothing(image) ? C_NULL : image

    API.lammps_create_atoms(lmp, numAtoms, id, types, x, v, image, bexpand ? 1 : 0)
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

function _string(ptr::Ptr)
    ptr == C_NULL && error("Wrapping NULL-pointer!")
    return Base.unsafe_string(ptr)
end

function _extract(ptr::Ptr{<:Real}, shape::Integer; copy=false, own=false)
    ptr == C_NULL && error("Wrapping NULL-pointer!")
    result = Base.unsafe_wrap(Array, ptr, shape; own=false)

    if own && copy
        result_copy = Base.copy(result)
        API.lammps_free(result)
        return result_copy
    end

    if own
        @static if VERSION >= v"1.11-dev"
            finalizer(API.lammps_free, result.ref.mem)
        else
            finalizer(API.lammps_free, result)
        end
    end

    return copy ? Base.copy(result) : result
end

function _extract(ptr::Ptr{<:Ptr{T}}, shape::NTuple{2}; copy=false) where T
    # The `own` kwarg is not implemented for 2D data, as this is currently not used anywhere

    ptr == C_NULL && error("Wrapping NULL-pointer!")

    prod(shape) == 0 && return Matrix{T}(undef, shape) # There is no data that can be wrapped

    # TODO: implement a debug mode to do this assert
    # pointers = Base.unsafe_wrap(Array, ptr, ndata)
    # @assert all(diff(pointers) .== count*sizeof(T))

    # If the pointers are evenly spaced, we can simply use the first pointer to wrap our matrix.
    first_pointer = unsafe_load(ptr)
    result = Base.unsafe_wrap(Array, first_pointer, shape; own=false)
    return copy ? Base.copy(result) : result
end

function _reinterpret(T::API._LMP_DATATYPE_CONST, ptr::Ptr)
    T === API.LAMMPS_INT && return Base.reinterpret(Ptr{Int32}, ptr)
    T === API.LAMMPS_INT_2D && return Base.reinterpret(Ptr{Ptr{Int32}}, ptr)
    T === API.LAMMPS_DOUBLE && return Base.reinterpret(Ptr{Float64}, ptr)
    T === API.LAMMPS_DOUBLE_2D && return Base.reinterpret(Ptr{Ptr{Float64}}, ptr)
    T === API.LAMMPS_INT64 && return Base.reinterpret(Ptr{Int64}, ptr)
    T === API.LAMMPS_INT64_2D && return Base.reinterpret(Ptr{Ptr{Int64}}, ptr)
    T === API.LAMMPS_STRING && return Base.reinterpret(Ptr{UInt8}, ptr)
end

_is_2D_datatype(lmp_dtype::API._LMP_DATATYPE_CONST) = lmp_dtype in (API.LAMMPS_INT_2D, API.LAMMPS_DOUBLE_2D, API.LAMMPS_INT64_2D)

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

        extract_setting(lmp, :dimension) |> println # 3
        extract_setting(lmp, :nlocal) |> println # 27
    end
```
"""
function extract_setting(lmp::LMP, name::Symbol)::Int32
    return API.lammps_extract_setting(lmp, name)
end

"""
    extract_global(lmp::LMP, name::String)

Extract a global property from a LAMMPS instance.
A full list of global variables can be found in the [lammps documentation](https://docs.lammps.org/Library_properties.html).
"""
Base.@constprop :aggressive function extract_global(lmp::LMP, name::Symbol)
    void_ptr = API.lammps_extract_global(lmp, name)
    void_ptr == C_NULL && throw(KeyError(name))

    lmp_type = extract_global_datatype(name)
    ptr = _reinterpret(lmp_type, void_ptr)

    lmp_type === API.LAMMPS_STRING && return _string(ptr)

    if name in (:boxlo, :boxhi, :sublo, :subhi, :sublo_lambda, :subhi_lambda, :periodicity, :procgrid)
        ptr = reinterpret(Ptr{NTuple{3, eltype(ptr)}}, ptr)
    elseif name in (:special_lj, :special_coul)
        ptr = reinterpret(Ptr{NTuple{4, eltype(ptr)}}, ptr)
    end

    return unsafe_load(ptr)
end

# marked as foldable to enable constant propagation
Base.@assume_effects :foldable function extract_global_datatype(name)
    # the handle get's ignored
    return API._LMP_DATATYPE_CONST(API.lammps_extract_global_datatype(C_NULL, name))
end

"""
    extract_atom(lmp::LMP, name::String; copy=false, with_ghosts=false)

Extract per-atom data from the lammps instance.


!!! info
    The returned data may become invalid if a re-neighboring operation
    is triggered at any point after calling this method. If this has happened,
    trying to read from this data will likely cause julia to crash.
    To prevent this, set `copy=true`.

A table with suported name keywords can be found in the [lammps documentation](https://docs.lammps.org/Classes_atom.html#_CPPv4N9LAMMPS_NS4Atom7extractEPKc).

## Arguments
- `copy`: determines whether lammps internal memory is used or if a copy is made.
- `with_ghosts`: Determines wheter entries for ghost atoms are included. This is ignored for "mass".
"""
Base.@constprop :aggressive function extract_atom(lmp::LMP, name::Symbol; copy=false, with_ghosts=false)
    void_ptr = API.lammps_extract_atom(lmp, name)
    void_ptr == C_NULL && throw(KeyError("Unknown per-atom variable $name"))

    lmp_type = extract_atom_datatype(name)
    ptr = _reinterpret(lmp_type, void_ptr)

    if name == :mass
        length = extract_global(lmp, :ntypes)
        ptr += sizeof(eltype(ptr)) # Scarry pointer arithemtic; The first entry in the array is unused
        return _extract(ptr, length; copy=copy)
    end

    length = extract_setting(lmp, with_ghosts ? :nall : :nlocal)

    if _is_2D_datatype(lmp_type)
        # only Quaternions have 4 entries
        # length is a Int32 and lammps_wrap expects a NTuple, so it's
        # neccecary to use Int32 for count as well
        count = name == "quat" ? Int32(4) : Int32(3)
        return _extract(ptr, (count, length); copy=copy)
    end

    return _extract(ptr, length; copy=copy)
end

Base.@assume_effects :foldable function extract_atom_datatype(name::Symbol)
    name_str = "$name"

    # custom properties; need to be handled seperately to ensure foldable
    startswith(name_str, "i_") && return API.LAMMPS_INT
    startswith(name_str, "i2_") && return API.LAMMPS_INT_2D
    startswith(name_str, "d_") && return API.LAMMPS_DOUBLE
    startswith(name_str, "d2_") && return API.LAMMPS_DOUBLE_2D

    # the datatypes of all build-in properties are compile time constants. Nevertheless, `extract_compute`
    # needs a valid handle. Therefore we use a dummy here.
    return API._LMP_DATATYPE_CONST(API.lammps_extract_atom_datatype(LAMMPS_DUMMY[], name))
end

"""
    extract_compute(lmp::LMP, name::String, style::_LMP_STYLE_CONST, lmp_type::_LMP_TYPE; copy::Bool=true)

Extract data provided by a compute command identified by the compute-ID.
Computes may provide global, per-atom, or local data, and those may be a scalar, a vector or an array.
Since computes may provide multiple kinds of data, it is required to set style and type flags representing what specific data is desired.

| valid values for `style`: |
| :------------------------ |
| `STYLE_GLOBAL`            |
| `STYLE_ATOM`              |
| `STYLE_LOCAL`             |

| valid values for `lmp_type`: | resulting return type: |
| :--------------------------- | :--------------------- |
| `TYPE_SCALAR`                | `Vector{Float64}`      |
| `TYPE_VECTOR`                | `Vector{Float64}`      |
| `TYPE_ARRAY`                 | `Matrix{Float64}`      |

Scalar values get returned as a vector with a single element. This way it's possible to
modify the internal state of the LAMMPS instance even if the data is scalar.

!!! info
    The returned data may become invalid as soon as another LAMMPS command has been issued at any point after calling this method.
    If this has happened, trying to read from this data will likely cause julia to crash.
    To prevent this, set `copy=true`.

# Examples

```julia
LMP(["-screen", "none"]) do lmp
    extract_compute(lmp, "thermo_temp", LMP_STYLE_GLOBAL, TYPE_VECTOR, copy=true)[2] = 2
    extract_compute(lmp, "thermo_temp", LMP_STYLE_GLOBAL, TYPE_VECTOR, copy=false)[3] = 3

    extract_compute(lmp, "thermo_temp", LMP_STYLE_GLOBAL, TYPE_SCALAR) |> println # [0.0]
    extract_compute(lmp, "thermo_temp", LMP_STYLE_GLOBAL, TYPE_VECTOR) |> println # [0.0, 0.0, 3.0, 0.0, 0.0, 0.0]
end
```
"""
function extract_compute(lmp::LMP, name::Symbol, style::_LMP_STYLE_CONST, lmp_type::_LMP_TYPE; copy::Bool=false)
    API.lammps_has_id(lmp, "compute", name) != 1 && throw(KeyError(name))

    void_ptr = API.lammps_extract_compute(lmp, name, style, get_enum(lmp_type))
    void_ptr == C_NULL && error("Compute $name doesn't have data matching $style, $(get_enum(lmp_type))")

    # `lmp_type in (SIZE_COLS, SIZE_ROWS, SIZE_VECTOR)` causes type instability for some reason
    if lmp_type == SIZE_COLS || lmp_type == SIZE_ROWS || lmp_type == SIZE_VECTOR
        ptr = _reinterpret(API.LAMMPS_INT, void_ptr)
        return unsafe_load(ptr)
    end

    if lmp_type == TYPE_SCALAR
        ptr = _reinterpret(API.LAMMPS_DOUBLE, void_ptr)
        return _extract(ptr, 1; copy=copy)
    end

    if lmp_type == TYPE_VECTOR
        ndata = (style == STYLE_ATOM) ?
            extract_setting(lmp, :nlocal) :
            extract_compute(lmp, name, style, SIZE_VECTOR)

        ptr = _reinterpret(API.LAMMPS_DOUBLE, void_ptr)
        return  _extract(ptr, ndata; copy=copy)
    end

    ndata = (style == STYLE_ATOM) ?
        extract_setting(lmp, :nlocal) :
        extract_compute(lmp, name, style, SIZE_ROWS)

    count = extract_compute(lmp, name, style, SIZE_COLS)
    ptr = _reinterpret(API.LAMMPS_DOUBLE_2D, void_ptr)

    return _extract(ptr, (count, ndata); copy=copy)
end

"""
    extract_variable(lmp::LMP, name::String, lmp_variable::LMP_VARIABLE, group::Union{String, Nothing}=nothing; copy::Bool=false)

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

the kwarg `copy` determies wheter a copy of the underlying data is made.
`copy` is only aplicable for `VAR_VECTOR` and `VAR_ATOM`. For all other variable types, a copy will be made regardless.
The underlying LAMMPS API call for `VAR_ATOM` internally allways creates a copy of the data. As the memory for this gets allocated by LAMMPS instead of julia,
it needs to be dereferenced using `LAMMPS.API.lammps_free` instead of through the garbage collector. 
If `copy=false` this gets acieved by registering `LAMMPS.API.lammps_free` as a finalizer for the returned data.
Alternatively, setting `copy=true` will instead create a new copy of the data. The lammps allocated block of memory will then be freed immediately.

the kwarg `group` determines for which atoms the variable will be extracted. It's only aplicable for
`VAR_ATOM` and will cause an error if used for other variable types. The entires for all atoms not in the group
will be zeroed out. By default, all atoms will be extracted.
"""
function extract_variable(lmp::LMP, name::Symbol, lmp_variable::_LMP_VARIABLE, group::Union{Symbol, Nothing}=nothing; copy::Bool=false)
    lmp_variable != VAR_ATOM && !isnothing(group) && throw(ArgumentError("the group parameter is only supported for per atom variables!"))

    if isnothing(group)
        group = C_NULL
    end

    void_ptr = API.lammps_extract_variable(lmp, name, group)
    void_ptr == C_NULL && throw(KeyError("Unknown variable $name"))

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
        ptr = _reinterpret(API.LAMMPS_DOUBLE, void_ptr)
        result = unsafe_load(ptr)
        API.lammps_free(ptr)
        return result
    end

    if lmp_variable == VAR_VECTOR
        # Calling lammps_extract_variable directly through the API instead of the higher level wrapper, as
        # "LMP_SIZE_VECTOR" is the only group name that won't be ignored for Vector Style Variables.
        # This isn't exposed to the high level API as it causes type instability for something that probably won't
        # ever be used outside of this implementation
        ndata_ptr = _reinterpret(API.LAMMPS_INT, API.lammps_extract_variable(lmp, name, "LMP_SIZE_VECTOR"))
        ndata = unsafe_load(ndata_ptr)
        API.lammps_free(ndata_ptr)

        ptr = _reinterpret(API.LAMMPS_DOUBLE, void_ptr)
        return _extract(ptr, ndata; copy=copy)
    end

    if lmp_variable == VAR_ATOM
        ndata = extract_setting(lmp, :nlocal)
        ptr = _reinterpret(API.LAMMPS_DOUBLE, void_ptr)
        # lammps expects us to take ownership of the data
        return _extract(ptr, ndata; copy=copy, own=true)
    end

    ptr = _reinterpret(API.LAMMPS_STRING, void_ptr)
    return _string(ptr)
end

function extract_variable_datatype(lmp::LMP, name)
    return API._LMP_VAR_CONST(API.lammps_extract_variable_datatype(lmp, name))
end

"""
    gather(lmp::LMP, name::Symbol; ids::Union{Nothing, Array{Int32}}=nothing, unwrap_image=false)

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
Base.@constprop :aggressive function gather(lmp::LMP, name::Symbol; ids::Union{Nothing, Array{Int32}}=nothing, unwrap_image=false)
    name == :mass && error("scattering/gathering mass is currently not supported! Use `extract_atom()` instead.")

    natoms = get_natoms(lmp)
    ndata = isnothing(ids) ? natoms : length(ids)

    count = _get_count(lmp, name)

    if is_fix_compute(name)
        data = iszero(count) ?
            Vector{Float64}(undef, ndata) :
            Matrix{Float64}(undef, count, ndata)
    elseif name == :image && unwrap_image
        count = 3
        data = Matrix{Int32}(undef, count, ndata)
    else
        lmp_type = extract_atom_datatype(name)
        if lmp_type == API.LAMMPS_DOUBLE
            data = Vector{Float64}(undef, count * ndata)
        elseif lmp_type == API.LAMMPS_DOUBLE_2D
            data = Matrix{Float64}(undef, count, ndata)
        elseif lmp_type == API.LAMMPS_INT
            data = Vector{Int32}(undef, count * ndata)
        elseif lmp_type == API.LAMMPS_INT_2D
            data = Matrix{Int32}(undef, count, ndata)
        end
    end

    dtype = eltype(data) == Float64
    count = max(1, count)

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
    scatter!(lmp::LMP, name::String, data::Array, ids::Union{Nothing, Array{Int32}}=nothing)

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
function scatter!(lmp::LMP, name::Symbol, data::Array; ids::Union{Nothing, Array{Int32}}=nothing)
    name == :mass && error("scattering/gathering mass is currently not supported! Use `extract_atom()` instead.")

    natoms = get_natoms(lmp)
    ndata = isnothing(ids) ? natoms : length(ids)
    count = _get_count(lmp, name)
    count = max(1, count)

    if name == :image && ndata*3 == length(data)
        count = 3
    end

    @assert count*ndata == length(data)

    if is_fix_compute(name)
        data::Array{Float64}
    else
        lmp_type = extract_atom_datatype(name)
        if lmp_type in (API.LAMMPS_DOUBLE, API.LAMMPS_DOUBLE_2D)
            data::Array{Float64}
        elseif lmp_type in (API.LAMMPS_INT, API.LAMMPS_INT_2D)
            data::Array{Int32}
        else
            throw(ArgumentError("key: $name does not have compatible datatype"))
        end
    end

    dtype = eltype(data) == Float64
    if isnothing(ids)
        API.lammps_scatter(lmp, name, dtype, count, data)
    else
        @assert all(1 <= id <= natoms for id in ids)
        API.lammps_scatter_subset(lmp, name, dtype, count, ndata, ids, data)
    end

    check(lmp)
end

Base.@assume_effects :foldable function is_fix_compute(name::Symbol)
    name_str = "$name"
    return startswith(name_str, r"[f,c]_")
end

function _get_count(lmp::LMP, name)
    # values taken from: https://docs.lammps.org/Classes_atom.html#_CPPv4N9LAMMPS_NS4Atom7extractEPKc

    name_str = "$name"
    if startswith(name_str, r"[f,c]_")
        if name_str[1] == 'c'
            count_ptr = API.lammps_extract_compute(lmp::LMP, name_str[3:end], API.LMP_STYLE_ATOM, API.LMP_SIZE_COLS)
        else
            count_ptr = API.lammps_extract_fix(lmp::LMP, name_str[3:end], API.LMP_STYLE_ATOM, API.LMP_SIZE_COLS, 0, 0)
        end
        count_ptr == C_NULL && throw(KeyError(name))
        count_ptr = _reinterpret(API.LAMMPS_INT, count_ptr)
        return unsafe_load(count_ptr)
    elseif name in (:mass, :id, :type, :mask, :image, :molecule, :q, :radius, :rmass, :ellipsoid, :line, :tri, :body, :temperature, :heatflow)
        return 1
    elseif name in (:x, :v, :f, :mu, :omega, :angmom, :torque)
        return 3
    elseif name == :quat
        return 4
    end
    throw(KeyError(name))
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
    ndata = extract_global(lmp, :nbonds)
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
    ndata = extract_global(lmp, :nangles)
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
    ndata = extract_global(lmp, :ndihedrals)
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
    ndata = extract_global(lmp, :nimpropers)[]
    data = Matrix{Int32}(undef, 5, ndata)
    API.lammps_gather_impropers(lmp, data)
    return data
end

"""
    group_to_atom_ids(lmp::LMP, group::String)

Find the IDs of the Atoms in the group.
"""
function group_to_atom_ids(lmp::LMP, group::Symbol)
    # Pad with '\0' to avoid confusion with groups names that are truncated versions of name
    # For example 'all' could be confused with 'a'
    name_padded = codeunits(String(group) * '\0')
    buffer_size = length(name_padded)
    buffer = zeros(UInt8, buffer_size)

    ngroups = API.lammps_id_count(lmp, :group)
    
    for idx in 0:ngroups-1
        API.lammps_id_name(lmp, :group, idx, buffer, buffer_size)
        buffer != name_padded && continue

        mask = gather(lmp, :mask) .& (1 << idx) .!= 0
        all_ids = UnitRange{Int32}(1, get_natoms(lmp))

        return all_ids[mask]
    end

    throw(KeyError(group))
end


"""
    get_category_ids(lmp::LMP, category::String, buffer_size::Integer=50)

Look up the names of entities within a certain category.

Valid categories are: compute, dump, fix, group, molecule, region, and variable.
names longer than `buffer_size` will be truncated to fit inside the buffer.
"""
function get_category_ids(lmp::LMP, category::Symbol, buffer_size::Integer=50)
    _check_valid_category(category)

    count = API.lammps_id_count(lmp, category)
    check(lmp)

    res = Vector{Symbol}(undef, count)

    for i in 1:count
        buffer = zeros(UInt8, buffer_size)
        API.lammps_id_name(lmp, category, i-1, buffer, buffer_size)
        res[i] = Symbol(buffer[1:findfirst(iszero, buffer)-1])
    end

    return res
end

_check_valid_category(category::Symbol) = category in (:compute, :dump, :fix, :group, :molecule, :region, :variable) || throw(KeyError(category))

end # module
