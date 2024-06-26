module LAMMPS
import MPI
include("api.jl")

export LMP, command, get_natoms, extract_atom, extract_compute, extract_global,
       gather, scatter!, group_to_atom_ids, get_category_ids

using Preferences

struct _LMP_DATATYPE{N}  end

const LAMMPS_INT = _LMP_DATATYPE{0}()
const LAMMPS_INT_2D = _LMP_DATATYPE{1}()
const LAMMPS_DOUBLE = _LMP_DATATYPE{2}()
const LAMMPS_DOUBLE_2D = _LMP_DATATYPE{3}()
const LAMMPS_INT64 = _LMP_DATATYPE{4}()
const LAMMPS_INT64_2D = _LMP_DATATYPE{5}()
const LAMMPS_STRING = _LMP_DATATYPE{6}()

struct _LMP_TYPE{N} end

const TYPE_SCALAR = _LMP_TYPE{0}()
const TYPE_VECTOR = _LMP_TYPE{1}()
const TYPE_ARRAY = _LMP_TYPE{2}()
const SIZE_VECTOR = _LMP_TYPE{3}()
const SIZE_ROWS = _LMP_TYPE{4}()
const SIZE_COLS = _LMP_TYPE{5}()

struct _LMP_STYLE{N} end

const STYLE_GLOBAL = _LMP_STYLE{0}()
const STYLE_ATOM = _LMP_STYLE{1}()
const STYLE_LOCAL = _LMP_STYLE{2}()

struct LMP_VARIABLE{N} end

const VARIABLE_EQUAL = LMP_VARIABLE{0}()
const VARIABLE_ATOM = LMP_VARIABLE{1}()
const VARIABLE_VECTOR = LMP_VARIABLE{2}()
const VARIABLE_STRING = LMP_VARIABLE{3}()

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
This constructor closes the LAMMPS instance immediately after `f` has executed.
"""
function LMP(f::Function, args=String[], comm=nothing)
    lmp = LMP(args, comm)
    result = f(lmp)
    close!(lmp)
    return result
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

function int2type(dtype)
    dtype == 0 && return LAMMPS_INT
    dtype == 1 && return LAMMPS_INT_2D
    dtype == 2 && return LAMMPS_DOUBLE
    dtype == 3 && return LAMMPS_DOUBLE_2D
    dtype == 4 && return LAMMPS_INT64
    dtype == 5 && return LAMMPS_INT64_2D
    dtype == 6 && return LAMMPS_STRING

    error("Unknown lammps data type: $dtype")
end

function type2julia(type::_LMP_DATATYPE)
    type == LAMMPS_INT && return Vector{Int32}
    type == LAMMPS_INT_2D && return Matrix{Int32}
    type == LAMMPS_DOUBLE && return Vector{Float64}
    type == LAMMPS_DOUBLE_2D && return Matrix{Float64}
    type == LAMMPS_INT64 && return Vector{Int64}
    type == LAMMPS_INT64_2D && return Matrix{Int64}
    type == LAMMPS_STRING && return String
end

function array2type(array::Union{VecOrMat, String})
    array isa Vector{Int32} && return LAMMPS_INT
    array isa Matrix{Int32} && return LAMMPS_INT_2D
    array isa Vector{Float64} && return LAMMPS_DOUBLE
    array isa Matrix{Float64} && return LAMMPS_DOUBLE_2D
    array isa Vector{Int64} && return LAMMPS_INT64
    array isa Matrix{Int64} && return LAMMPS_INT64_2D
    array isa String && return LAMMPS_STRING
end

is_2D(N::Integer) = N in (1, 3, 5)
is_2D(::_LMP_DATATYPE{N}) where N = N in (1, 3, 5)
Base.Int(::_LMP_DATATYPE{N}) where N = N
Base.Int(::LMP_VARIABLE{N}) where N = N
Base.Int(::_LMP_TYPE{N}) where N = N
Base.Int(::_LMP_STYLE{N}) where N = N

function lammps_reinterpret(T::_LMP_DATATYPE, ptr::Ptr)
    # we're pretty much guaranteed to call lammps_reinterpret after reciving a pointer
    # from LAMMPS. So this is a good spot catch NULL-pointers and avoid Segfaults
    ptr == C_NULL && error("reinterpreting NULL-pointer!")

    T === LAMMPS_INT && return Base.reinterpret(Ptr{Int32}, ptr)
    T === LAMMPS_INT_2D && return Base.reinterpret(Ptr{Ptr{Int32}}, ptr)
    T === LAMMPS_DOUBLE && return Base.reinterpret(Ptr{Float64}, ptr)
    T === LAMMPS_DOUBLE_2D && return Base.reinterpret(Ptr{Ptr{Float64}}, ptr)
    T === LAMMPS_INT64 && return Base.reinterpret(Ptr{Int64}, ptr)
    T === LAMMPS_INT64_2D && return Base.reinterpret(Ptr{Ptr{Int64}}, ptr)
    T === LAMMPS_STRING && return Base.reinterpret(Ptr{UInt8}, ptr)
end

"""
    extract_global(lmp::LMP, name::String, dtype::_LMP_DATATYPE; copy=true)
"""
function extract_global(lmp::LMP, name::String, dtype::_LMP_DATATYPE; copy=true)
    @assert API.lammps_extract_global_datatype(lmp, name) == Int(dtype)

    ptr = lammps_reinterpret(dtype, API.lammps_extract_global(lmp, name))

    dtype == LAMMPS_STRING && return lammps_unsafe_string(ptr, copy)

    if name in ("boxlo", "boxhi", "sublo", "subhi", "sublo_lambda", "subhi_lambda", "periodicity")
        length = 3
    elseif name in ("special_lj", "special_coul")
        length = 4
    else
        length = 1
    end

    return lammps_unsafe_wrap(ptr, length, copy)
end

function lammps_unsafe_string(ptr::Ptr, copy=true)
    result = Base.unsafe_string(ptr)
    return copy ? deepcopy(result) : result
end

function lammps_unsafe_wrap(ptr::Ptr{<:Real}, shape::Integer, copy=true)
    result = Base.unsafe_wrap(Array, ptr, shape, own=own)
    return copy ? Base.copy(result) : result
end

function lammps_unsafe_wrap(ptr::Ptr{<:Ptr{T}}, shape::NTuple{2}, copy=true) where T
    (count, ndata) = shape

    ndata == 0 && return Matrix{T}(undef, count, ndata)

    pointers = Base.unsafe_wrap(Array, ptr, ndata)

    @assert all(diff(pointers) .== count*sizeof(T))
    result = Base.unsafe_wrap(Array, pointers[1], shape, own=own)

    return copy ? Base.copy(result) : result
end

"""
    extract_setting(lmp::LMP, name::String)

<https://docs.lammps.org/Library_properties.html#_CPPv422lammps_extract_settingPvPKc>
"""
function extract_setting(lmp::LMP, name::String)
    return API.lammps_extract_setting(lmp, name)
end

"""
    extract_atom(lmp::LMP, name::String, dtype::_LMP_DATATYPE; copy=true)
"""
function extract_atom(lmp::LMP, name::String, dtype::_LMP_DATATYPE; copy=true)
    @assert API.lammps_extract_atom_datatype(lmp, name) == Int(dtype)

    ptr = lammps_reinterpret(dtype, API.lammps_extract_atom(lmp, name))
    @assert ptr != C_NULL

        if name == "mass"
        length = extract_global(lmp, "ntypes", LAMMPS_INT, copy=false)[]
        ptr += sizeof(Float64) # Scarry pointer arithemtic
        result = lammps_unsafe_wrap(ptr, length, false)

        return copy ? Base.copy(result) : result
    end

    length = extract_setting(lmp, "nlocal")

    if is_2D(dtype)
        count = name == "quat" ? Int32(4) : Int32(3) # only Quaternions have 4 entries
        return lammps_unsafe_wrap(ptr, (count, length), copy)
    end

    return lammps_unsafe_wrap(ptr, length, copy)
end

"""
    extract_compute(lmp::LMP, name::String, style::_LMP_STYLE, type::_LMP_TYPE; copy=true)
"""
function extract_compute(lmp::LMP, name::String, style::_LMP_STYLE, type::_LMP_TYPE; copy=true)
    void_ptr = API.lammps_extract_compute(lmp, name, Int(style), Int(type))
    @assert void_ptr != C_NULL

    if type in (SIZE_COLS, SIZE_ROWS, SIZE_VECTOR)
        ptr = lammps_reinterpret(LAMMPS_INT, void_ptr)
        return lammps_unsafe_wrap(ptr, 1, copy)
    end

    if type == TYPE_SCALAR
        ptr = lammps_reinterpret(LAMMPS_DOUBLE, void_ptr)
        return lammps_unsafe_wrap(ptr, 1, copy)
    end

    ndata = style == STYLE_ATOM ?
        extract_setting(lmp, "nlocal") :
        extract_compute(lmp, name, style, TYPE_SCALAR, copy=false)[]

    if type == TYPE_VECTOR
        ptr = lammps_reinterpret(LAMMPS_DOUBLE, void_ptr)
        return lammps_unsafe_wrap(ptr, ndata, copy)
    end

    count = extract_compute(lmp, name, style, SIZE_COLS)[]
    ptr = lammps_reinterpret(LAMMPS_DOUBLE_2D, void_ptr)

    return lammps_unsafe_wrap(ptr, (count, ndata), copy)
end

"""
    extract_variable(lmp::LMP, name::String, variable::LMP_VARIABLE, group=C_NULL; copy=true)

Extracts the data from a LAMMPS variable. When the variable is either an `equal`-style compatible variable,
a `vector`-style variable, or an `atom`-style variable, the variable is evaluated and the corresponding value(s) returned.
Variables of style `internal` are compatible with `equal`-style variables, if they return a numeric value.
For other variable styles, their string value is returned.
"""
function extract_variable(lmp::LMP, name::String, variable::LMP_VARIABLE, group=C_NULL; copy=true)
    @assert variable == VARIABLE_ATOM || group == C_NULL "the group parameter is only supported for per atom variables!"
    @assert API.lammps_extract_variable_datatype(lmp, name) == Int(variable)

    void_ptr = API.lammps_extract_variable(lmp, name, group)
    @assert void_ptr != C_NULL

    if variable == VARIABLE_EQUAL
        ptr = lammps_reinterpret(LAMMPS_DOUBLE, void_ptr)
        result = unsafe_load(ptr)
        API.lammps_free(ptr)
        return result
    end

    if variable == VARIABLE_VECTOR
        ndata_ptr = lammps_reinterpret(LAMMPS_INT, API.lammps_extract_variable(lmp, name, "GET_VECTOR_SIZE"))
        ndata = unsafe_load(ndata_ptr)
        API.lammps_free(ndata_ptr)

        ptr = lammps_reinterpret(LAMMPS_DOUBLE, void_ptr)
        return lammps_unsafe_wrap(ptr, ndata, copy)
    end

    if variable == VARIABLE_ATOM
        ndata = extract_setting(lmp, "nlocal")

        ptr = lammps_reinterpret(LAMMPS_DOUBLE, void_ptr)
        result = lammps_unsafe_wrap(ptr, ndata, true)
        API.lammps_free(ptr)
        return result
    end

    ptr = lammps_reinterpret(LAMMPS_STRING, void_ptr)
    return lammps_unsafe_string(ptr, copy)
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
function gather(lmp::LMP, name::String, T::_LMP_DATATYPE, ids::Union{Nothing, Array{Int32}}=nothing)
    name == "mass" && error("scattering/gathering mass is currently not supported! Use `extract_atom()` instead.")

    count = _get_count(lmp, name)
    _dtype = _get_dtype(lmp, name)

    @assert Int(T) in _dtype "Expected data type $(int2type.(_dtype)) got $T instead."
    count > 1 && T in (LAMMPS_DOUBLE, LAMMPS_INT) && error("1")

    dtype = T in (LAMMPS_DOUBLE, LAMMPS_DOUBLE_2D)
    natoms = get_natoms(lmp)
    ndata = isnothing(ids) ? natoms : length(ids)

    if T == LAMMPS_INT
        data = Vector{Int32}(undef, ndata)
    elseif T == LAMMPS_DOUBLE
        data = Vector{Int32}(undef, ndata)
    elseif T == LAMMPS_INT_2D
        data = Matrix{Float64}(undef, (count, ndata))
    elseif T == LAMMPS_DOUBLE_2D
        data = Matrix{Float64}(undef, (count, ndata))
    else
        error("2")
    end

    if isnothing(ids)
        API.lammps_gather(lmp, name, dtype, count, data)
    else
        @assert all(1 <= id <= natoms for id in ids)
        API.lammps_gather_subset(lmp, name, dtype, count, ndata, ids, data)
    end

    check(lmp)
    return data
end

function gather_bonds(lmp::LMP)
    nbonds = extract_global(lmp, "nbonds", LAMMPS_INT64, copy=false)[]
    data = Matrix{Int32}(undef, (3, nbonds))
    API.lammps_gather_bonds(lmp, data)
    return data
end

function gather_angles(lmp::LMP)
    nangles = extract_global(lmp, "nangles", LAMMPS_INT64, copy=false)[]
    data = Matrix{Int32}(undef, (4, nangles))
    API.lammps_gather_angles(lmp, data)
    return data
end

function gather_dihedrals(lmp::LMP)
    ndihedrals = extract_global(lmp, "ndihedrals", LAMMPS_INT64, copy=false)[]
    data = Matrix{Int32}(undef, (5, ndihedrals))
    API.lammps_gather_dihedrals(lmp, data)
    return data
end

function gather_impropers(lmp::LMP)
    nimpropers = extract_global(lmp, "nimpropers", LAMMPS_INT64, copy=false)[]
    data = Matrix{Int32}(undef, (5, nimpropers))
    API.lammps_gather_impropers(lmp, data)
    return data
end

function create_atoms(lmp::LMP, type, x; id=nothing, v=nothing, image=nothing, bexpand=false)
    natoms = length(type)

    @assert size(x) == (3, natoms)

    isnothing(id) ? id = C_NULL : @assert size(id) == natoms
    isnothing(v) ? v = C_NULL : @assert size(v) == (3, natoms)
    isnothing(image) ? image = C_NULL : @assert size(image) = natoms

    return API.lammps_create_atoms(lmp, natoms, id, type, x, v, image, bexpand)
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
function scatter!(lmp::LMP, name::String, data::T, ids::Union{Nothing, Array{Int32}}=nothing) where T<:VecOrMat
    name == "mass" && error("scattering/gathering mass is currently not supported! Use `extract_atom()` instead.")

    count = _get_count(lmp, name)
    _T = _get_dtype(lmp, name)

    @assert Int(array2type(T)) == _T

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
        count_ptr == C_NULL && error("compute $name does not have per atom data")
        count_ptr = lammps_reinterpret(LAMMPS_INT, count_ptr)
        count = unsafe_load(count_ptr)
    
        # a count of 0 indicates that the entity is a vector. In order to perserve type stability we just treat that as a 1xN Matrix.
        return count == 0 ? 1 : count
    else
        dtype = API.lammps_extract_atom_datatype(lmp, name)
        dtype == -1 && error("Unkown per atom property $name")

        name == "quat" && return 4
        is_2D(dtype) && return 3
        return 1

    end
end

function _get_dtype(lmp::LMP, name::String)
    if startswith(name, r"[f,c]_")
        return (Int(LAMMPS_DOUBLE), Int(LAMMPS_DOUBLE_2D))
    else
        return (API.lammps_extract_atom_datatype(lmp, name), )
    end
end

function decode_image_flags(images::Vector{<:Integer})
    data = Matrix{Int32}(undef, (3, length(images)))

    for (i, image) in pairs(images)
        data_view = @view data[:, i]
        API.lammps_decode_image_flags(image, data_view)
    end

    return data
end

function encode_image_flags(images::Matrix{<:Integer})
    @assert size(images, 1) == 3

    return [API.lammps_encode_image_flags(img[1], img[2], img[3]) for img in eachcol(images)]
end

function is_running(lmp)
    return API.lammps_is_running(lmp) > 0
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

end # module
