module LAMMPS
import MPI
using LinearAlgebra
import OpenBLAS32_jll
import StaticArrays: SVector, SMatrix, MVector, SA
import DifferentiationInterface: AbstractADType, Constant, value_and_derivative
import Bumper: @no_escape, @alloc
import UnsafeArrays: UnsafeArray

include("api.jl")

export LMP, command, create_atoms, get_natoms, extract_atom, extract_compute, extract_global,
       extract_setting, extract_box, reset_box, gather, gather_bonds, gather_angles, gather_dihedrals,
       gather_impropers, scatter!, group_to_atom_ids, get_category_ids, extract_variable, LAMMPSError, FixExternal,
       PairExternal, set_energy_peratom, set_virial_peratom,
       encode_image_flags, decode_image_flags, compute_neighborlist, fix_neighborlist, pair_neighborlist,
       get_mpi_comm,

       # _LMP_DATATYPE
       LAMMPS_NONE,
       LAMMPS_INT,
       LAMMPS_INT_2D,
       LAMMPS_DOUBLE,
       LAMMPS_DOUBLE_2D,
       LAMMPS_INT64,
       LAMMPS_INT64_2D,
       LAMMPS_STRING,

       # _LMP_TYPE
       TYPE_SCALAR,
       TYPE_VECTOR,
       TYPE_ARRAY,
       SIZE_VECTOR,
       SIZE_ROWS,
       SIZE_COLS,

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

const BIGINT = API.lammps_extract_setting(C_NULL, "bigint") == 4 ? Int32 : Int64
const TAGINT = API.lammps_extract_setting(C_NULL, "tagint") == 4 ? Int32 : Int64
const IMAGEINT = API.lammps_extract_setting(C_NULL, "imageint") == 4 ? Int32 : Int64

function __init__()
    # LAMMPS requires using LP64, default to OpenBLAS32 if not already available
    config = LinearAlgebra.BLAS.lbt_get_config()
    if !any(lib -> lib.interface == :lp64, config.loaded_libs)
        LinearAlgebra.BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path)
    end

    BIGINT != (API.lammps_extract_setting(C_NULL, "bigint") == 4 ? Int32 : Int64) &&
        error("The size of the LAMMPS integer type BIGINT has changed! To fix this, you need to manually invalidate the LAMMPS.jl cache.")
    TAGINT != (API.lammps_extract_setting(C_NULL, "tagint") == 4 ? Int32 : Int64) &&
         error("The size of the LAMMPS integer type TAGINT has changed! To fix this, you need to manually invalidate the LAMMPS.jl cache.")
     IMAGEINT != (API.lammps_extract_setting(C_NULL, "tagint") == 4 ? Int32 : Int64) &&
         error("The size of the LAMMPS integer type IMAGEINT has changed! To fix this, you need to manually invalidate the LAMMPS.jl cache.")

    if API.lammps_config_has_mpi_support() == 0
        @warn "The currently loaded LAMMPS installation does not have MPI enabled! \n" *
        "Please provide your own LAMMPS installation with `LAMMPS.set_library()` if you \n" *
        "want to run LAMMPS in parallel using MPI." 
    end

    if API.lammps_config_has_exceptions() == 0
        @warn "The currently loaded LAMMPS installation doesn't have exceptions enabled! \n" *
        "This causes the REPL to crash whenever LAMMPS encounters an error."
    end
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
    LMP(args::Vector{String}=String[], comm::MPI.Comm=MPI.COMM_WORLD)

Create a new LAMMPS instance while passing in a list of strings as if they were command-line arguments for the LAMMPS executable.

A full ist of command-line options can be found in the [lammps documentation](https://docs.lammps.org/Run_options.html).
"""
mutable struct LMP
    @atomic handle::Ptr{Cvoid}
    external_fixes::Dict{String, Any}

    function LMP(args::Vector{String}=String[], comm::MPI.Comm=MPI.COMM_WORLD)
        args = copy(args)
        pushfirst!(args, "lammps")

        GC.@preserve args begin
            if API.lammps_config_has_mpi_support() == 0
                handle = API.lammps_open_no_mpi(length(args), args, C_NULL)
            else
                if !MPI.Initialized()
                    error("MPI has not been initialized. Make sure to first call `MPI.Init()`")
                end
                handle = API.lammps_open(length(args), args, comm, C_NULL)
            end
        end

        if API.lammps_has_error(handle) != 0
            buf = zeros(UInt8, 100)
            API.lammps_get_last_error_message(handle, buf, length(buf))
            msg = replace(rstrip(String(buf), '\0'), "ERROR: " => "")
            throw(LAMMPSError(msg))
        end

        this = new(handle, Dict{String, Any}())
        finalizer(close!, this)

        ver = version(this)
        if ver < 20250402 
            loaded = string(ver)[1:4] * '-' * string(ver)[5:6] * '-' * string(ver)[7:8]
            error("This version of LAMMPS.jl is only compatible with lammps version 2025-04-02 or newer.\nThe currently loaded version of lammps is $loaded")
        end
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
        empty!(lmp.external_fixes)
        API.lammps_close(handle)
    end
    return nothing
end

"""
    LMP(f::Function, args=String[], comm=MPI.COMM_WORLD)

Create a new LAMMPS instance and call `f` on that instance while returning the result from `f`.
"""
function LMP(f::Function, args=String[], comm=MPI.COMM_WORLD)
    lmp = LMP(args, comm)
    return f(lmp)
    # `close!` is registered as a finalizer for LMP, no need to close it here.
end

function version(lmp::LMP)
    API.lammps_version(lmp)
end

"""
    get_mpi_comm(lmp::LMP)::Union{Nothing, MPI.Comm}

Return the MPI communicator used by the lammps instance or `nothing` if the lammps build doesn't support MPI.
"""
function get_mpi_comm(lmp::LMP)::Union{Nothing, MPI.Comm}
    comm_f = API.lammps_get_mpi_comm(lmp)
    comm_f == -1 && return nothing
    comm_c = MPI.API.MPI_Comm_f2c(comm_f)
    return MPI.Comm(comm_c)
end

struct LAMMPSError <: Exception
    msg::String
end

function LAMMPSError(lmp::LMP)
    buf = zeros(UInt8, 255)
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
    encode_image_flags(ix, iy, iz)
    encode_image_flags(flags)

Encode three integer image flags into a single imageint.
"""
encode_image_flags(ix, iy, iz) = API.lammps_encode_image_flags(ix, iy, iz)
encode_image_flags(flags) = API.lammps_encode_image_flags(flags...)

"""
    decode_image_flags(image)

Decode a single image flag integer into three regular integers.
"""
function decode_image_flags(image)
    flags = Ref{NTuple{3, Cint}}()
    @inline API.lammps_decode_image_flags(image, flags)
    return flags[]
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


@deprecate gather_atoms(lmp::LMP, name, T, count) gather(lmp, name, T)


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

struct NeighListVec <: AbstractVector{Cint}
    numneigh::Int
    neighbors::Ptr{Int32}
end

function Base.getindex(nle::NeighListVec, i::Integer)
    @boundscheck checkbounds(nle, i)
    return unsafe_load(nle.neighbors, i)+Cint(1)
end

Base.size(nle::NeighListVec) = (nle.numneigh,)

struct NeighList <: AbstractVector{Pair{Int32, NeighListVec}}
    lmp::LMP
    idx::Cint
end

function Base.getindex(nl::NeighList, element::Integer)
    iatom = Ref{Cint}()
    numneigh = Ref{Cint}()
    neighbors = Ref{Ptr{Cint}}()
    @inline API.lammps_neighlist_element_neighbors(nl.lmp, nl.idx, element-one(element) #= 0-based indexing =#, iatom, numneigh, neighbors)
    @boundscheck iatom[] == -1 && throw(BoundsError(nl, element))
    return iatom[]+Cint(1) => NeighListVec(numneigh[], neighbors[])
end

Base.size(nl::NeighList) = (API.lammps_neighlist_num_elements(nl.lmp, nl.idx),)

"""
    compute_neighborlist(lmp::LMP, id::String; request=0)

Retrieve neighbor list requested by a compute.

The neighbor list request from a compute is identified by the compute ID and the request ID.
The request ID is typically 0, but will be > 0 in case a compute has multiple neighbor list requests.

Each neighbor list contains vectors of local indices of neighboring atoms.
These can be used to index into Arrays returned from `extract_atom`.
"""
function compute_neighborlist(lmp::LMP, id::String; request=0)
    idx = API.lammps_find_compute_neighlist(lmp, id, request)
    idx == -1 && throw(KeyError(id))
    return NeighList(lmp, idx)
end

"""
    fix_neighborlist(lmp::LMP, id::String; request=0)

Retrieve neighbor list requested by a fix.

The neighbor list request from a fix is identified by the fix ID and the request ID.
The request ID is typically 0, but will be > 0 in case a fix has multiple neighbor list requests.

Each neighbor list contains vectors of local indices of neighboring atoms.
These can be used to index into Arrays returned from `extract_atom`.
"""
function fix_neighborlist(lmp::LMP, id::String; request=0)
    idx = API.lammps_find_compute_neighlist(lmp, id, request)
    idx == -1 && throw(KeyError(id))
    return NeighList(lmp, idx)
end

"""
    pair_neighborlist(lmp::LMP, style::String; exact=false, nsub=0, request=0)

This function determines which of the available neighbor lists for pair styles matches the given conditions. It first matches the style name. If exact is true the name must match exactly,
if exact is false, a regular expression or sub-string match is done. If the pair style is hybrid or hybrid/overlay the style is matched against the sub styles instead.
If a the same pair style is used multiple times as a sub-style, the nsub argument must be > 0 and represents the nth instance of the sub-style (same as for the pair_coeff command, for example).
In that case nsub=0 will not produce a match and this function will Error.

The final condition to be checked is the request ID (reqid). This will normally be 0, but some pair styles request multiple neighbor lists and set the request ID to a value > 0.

Each neighbor list contains vectors of local indices of neighboring atoms.
These can be used to index into Arrays returned from `extract_atom`.

## Examples
```julia
lmp = LMP()

command(lmp, \"""
    region cell block 0 3 0 3 0 3
    create_box 1 cell
    lattice sc 1
    create_atoms 1 region cell
    mass 1 1

    pair_style zero 1.0
    pair_coeff * *

    run 1
\""")

x = extract_atom(lmp, "x", LAMMPS_DOUBLE_2D; with_ghosts=true)

for (iatom, neighs) in pair_neighborlist(lmp, "zero")
    for jatom in neighs
        ix = @view x[:, iatom]
        jx = @view x[:, jatom]

        println(ix => jx)
   end
end
```
"""
function pair_neighborlist(lmp::LMP, style::String; exact=false, nsub=0, request=0)
    idx = API.lammps_find_pair_neighlist(lmp, style, exact, nsub, request)
    idx == -1 && throw(KeyError(style))
    return NeighList(lmp, idx)
end


include("external.jl")

end # module
