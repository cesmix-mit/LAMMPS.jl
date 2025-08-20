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
       extract_setting, extract_box, reset_box, gather, gather!, gather_bonds, gather_angles, gather_dihedrals,
       gather_impropers, scatter!, group_to_atom_ids, get_category_ids, extract_variable, LAMMPSError, FixExternal,
       PairExternal, set_energy!, set_virial!, InteractionConfig,
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

        mask = gather(lmp, "mask", LAMMPS_INT) .& (1 << idx) .!= 0
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

include("extract.jl")
include("gather_scatter.jl")
include("neighborlist.jl")
include("external.jl")

end # module
