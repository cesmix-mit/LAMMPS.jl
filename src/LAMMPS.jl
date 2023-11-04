module LAMMPS
import MPI
include("api.jl")

export LMP, command, get_natoms, extract_atom, extract_compute, extract_global,
       gather_atoms

using Preferences

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

mutable struct LMP
    @atomic handle::Ptr{Cvoid}
    external_fixes::Dict{String, Any}

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

        this = new(handle, Dict{String, Any}())
        finalizer(close!, this)
        return this
    end
end
Base.unsafe_convert(::Type{Ptr{Cvoid}}, lmp::LMP) = lmp.handle

"""
    close!(lmp::LMP)

Shutdown an LMP instance.
"""
function close!(lmp::LMP)
    handle = @atomicswap lmp.handle = C_NULL
    if handle !== C_NULL 
        empty!(lmp.external_fixes)
        API.lammps_close(handle)
    end
    return nothing
end

function LMP(f::Function, args=String[], comm=nothing)
    lmp = LMP(args, comm)
    f(lmp)
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
        error(String(buf))
    end
end


"""
    command(lmp::lmp, cmd)
"""
function command(lmp::LMP, cmd)
    ptr = API.lammps_command(lmp, cmd)
    ptr == C_NULL && check(lmp)
    nothing
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

function dtype2type(dtype::API._LMP_DATATYPE_CONST)
    if dtype == API.LAMMPS_INT
        type = Ptr{Int32}
    elseif dtype == API.LAMMPS_INT_2D
        type = Ptr{Ptr{Int32}}
    elseif dtype == API.LAMMPS_INT64
        type = Ptr{Int64}
    elseif dtype == API.LAMMPS_INT64_2D
        type = Ptr{Ptr{Int64}}
    elseif dtype == API.LAMMPS_DOUBLE
        type = Ptr{Float64}
    elseif dtype == API.LAMMPS_DOUBLE_2D
        type = Ptr{Ptr{Float64}}
    elseif dtype == API.LAMMPS_STRING
        type = Ptr{Cchar}
    else
        @assert false "Unknown dtype: $dtype"
    end
    return type
end

"""
    extract_global(lmp, name, dtype=nothing)
"""
function extract_global(lmp::LMP, name, dtype=nothing)
    if dtype === nothing
        dtype = API.lammps_extract_global_datatype(lmp, name)
        dtype == -1 && error("Could not find dataype for global $name")
    end
    dtype = API._LMP_DATATYPE_CONST(dtype)
    type = dtype2type(dtype)

    ptr = API.lammps_extract_global(lmp, name)
    ptr = reinterpret(type, ptr)

    if ptr !== C_NULL
        if dtype == API.LAMMPS_STRING
            return Base.unsafe_string(ptr)
        end
        # TODO: deal with non-scalar data
        return Base.unsafe_load(ptr)
    end
end

function unsafe_wrap(ptr, shape)
    if length(shape) > 1
        # We got a list of ptrs,
        # but the first pointer points to the whole data
        ptr = Base.unsafe_load(ptr)

        @assert length(shape) == 2

        # Note: Julia like Fortran is column-major
        #       so the data is transposed from Julia's perspective
        shape = reverse(shape)
    end

    # TODO: Who is responsible for freeing this data
    array = Base.unsafe_wrap(Array, ptr, shape, own=false)
    return array
end

"""
    extract_atom(lmp, name, dtype=nothing, axes1, axes2)
"""
function extract_atom(lmp::LMP, name,
                      dtype::Union{Nothing, API._LMP_DATATYPE_CONST} = nothing,
                      axes1=nothing, axes2=nothing)


    if dtype === nothing
        dtype = API.lammps_extract_atom_datatype(lmp, name)
        dtype == -1 && error("Could not find dataype for atom $name")
        dtype = API._LMP_DATATYPE_CONST(dtype)
    end

    if axes1 === nothing
        if name == "mass"
            axes1 = extract_global(lmp, "ntypes") + 1
        else
            axes1 = extract_global(lmp, "nlocal") % Int
        end
    end

    if axes2 === nothing
        if dtype in (API.LAMMPS_INT_2D, API.LAMMPS_INT64_2D, API.LAMMPS_DOUBLE_2D)
            # TODO: Other fields?
            if name in ("x", "v", "f", "angmom", "torque", "csforce", "vforce")
                axes2 = 3
            else
                axes2 = 2
            end
        end
    end

    if axes2 !== nothing
        shape = (axes1, axes2)
    else
        shape = (axes1, )
    end

    type = dtype2type(dtype)
    ptr = API.lammps_extract_atom(lmp, name)
    ptr = reinterpret(type, ptr)

    unsafe_wrap(ptr, shape)
end

function unsafe_extract_compute(lmp::LMP, name, style, type)
    if type == API.LMP_TYPE_SCALAR
        if style == API.LMP_STYLE_GLOBAL
            dtype = Ptr{Float64}
        elseif style == API.LMP_STYLE_LOCAL
            dtype = Ptr{Cint}
        elseif style == API.LMP_STYLE_ATOM
            return nothing
        end
        extract = true
    elseif type == API.LMP_TYPE_VECTOR
        dtype = Ptr{Float64}
        extract = false
    elseif type == API.LMP_TYPE_ARRAY
        dtype = Ptr{Ptr{Float64}}
        extract = false
    elseif type == API.LMP_SIZE_COLS
        dtype = Ptr{Cint}
        extract = true
    elseif type == API.LMP_SIZE_ROWS ||
           type == API.LMP_SIZE_VECTOR
        if style == API.LMP_STYLE_ATOM
            return nothing
        end
        dtype = Ptr{Cint}
        extract = true
    else
        @assert false "Unknown type: $type"
    end

    ptr = API.lammps_extract_compute(lmp, name, style, type)
    ptr == C_NULL && check(lmp)

    if ptr == C_NULL
        error("Could not extract_compute $name with $style and $type")
    end

    ptr = reinterpret(dtype, ptr)
    if extract
        return Base.unsafe_load(ptr)
    end
    return ptr
end

"""
    extract_compute(lmp, name, style, type)
"""
function extract_compute(lmp::LMP, name, style, type)
    ptr_or_value = unsafe_extract_compute(lmp, name, style, type)
    if style == API.LMP_TYPE_SCALAR
        return ptr_or_value
    end
    if ptr_or_value === nothing
        return nothing
    end
    ptr = ptr_or_value::Ptr

    if style in (API.LMP_STYLE_GLOBAL, API.LMP_STYLE_LOCAL)
        if type == API.LMP_TYPE_VECTOR
            nrows = unsafe_extract_compute(lmp, name, style, API.LMP_SIZE_VECTOR)
            return unsafe_wrap(ptr, (nrows,))
        elseif type == API.LMP_TYPE_ARRAY
            nrows = unsafe_extract_compute(lmp, name, style, API.LMP_SIZE_ROWS)
            ncols = unsafe_extract_compute(lmp, name, style, API.LMP_SIZE_COLS)
            return unsafe_wrap(ptr, (nrows, ncols))
        end
    else style = API.LMP_STYLE_ATOM
        nlocal = extract_global(lmp, "nlocal")
        if type == API.LMP_TYPE_VECTOR
            return unsafe_wrap(ptr, (nlocal,))
        elseif type == API.LMP_TYPE_ARRAY
            ncols = unsafe_extract_compute(lmp, name, style, API.LMP_SIZE_COLS)
            return unsafe_wrap(ptr, (nlocal, ncols))
        end
    end
    return nothing
end

"""
    extract_variable(lmp::LMP, name, group)

Extracts the data from a LAMMPS variable. When the variable is either an `equal`-style compatible variable,
a `vector`-style variable, or an `atom`-style variable, the variable is evaluated and the corresponding value(s) returned.
Variables of style `internal` are compatible with `equal`-style variables, if they return a numeric value.
For other variable styles, their string value is returned.
"""
function extract_variable(lmp::LMP, name::String, group=nothing)
    var = API.lammps_extract_variable_datatype(lmp, name)
    if var == -1
        throw(KeyError(name))
    end
    if group === nothing
        group = C_NULL
    end

    if var == API.LMP_VAR_EQUAL
        ptr = API.lammps_extract_variable(lmp, name, C_NULL)
        val = Base.unsafe_load(Base.unsafe_convert(Ptr{Float64}, ptr))
        API.lammps_free(ptr)
        return val
    elseif var == API.LMP_VAR_ATOM
        nlocal = extract_global(lmp, "nlocal")
        ptr = API.lammps_extract_variable(lmp, name, group)
        if ptr == C_NULL
            error("Group $group for variable $name with style atom not available.")
        end
        # LAMMPS uses malloc, so and we are taking ownership of this buffer
        val = copy(Base.unsafe_wrap(Array, Base.unsafe_convert(Ptr{Float64}, ptr), nlocal; own=false))
        API.lammps_free(ptr)
        return val
    elseif var == API.LMP_VAR_VECTOR
        # TODO Fix lammps docs `GET_VECTOR_SIZE`
        ptr = API.lammps_extract_variable(lmp, name, "LMP_SIZE_VECTOR")
        if ptr == C_NULL
            error("$name is a vector style variable but has no size.")
        end
        sz = Base.unsafe_load(Base.unsafe_convert(Ptr{Cint}, ptr))
        API.lammps_free(ptr)
        ptr = API.lammps_extract_variable(lmp, name, C_NULL)
        return Base.unsafe_wrap(Array, Base.unsafe_convert(Ptr{Float64}, ptr), sz, own=false)
    elseif var == API.LMP_VAR_STRING
        ptr = API.lammps_extract_variable(lmp, name, C_NULL)
        return Base.unsafe_string(Base.unsafe_convert(Ptr{Cchar}, ptr))
    else
        error("Unkown variable style $var")
    end
end

function extract_setting(lmp, name)
    val = API.lammps_extract_setting(lmp, name)
    val == -1 && error("Could not find setting $name")
    return val
end

function gather_atoms(lmp::LMP, name, T, count)
    if T === Int32
        dtype = 0
    elseif T === Float64
        dtype = 1
    else
        error("Only Int32 or Float64 allowed as T, got $T")
    end
    natoms = get_natoms(lmp)
    data = Array{T, 2}(undef, (count, natoms))
    API.lammps_gather_atoms(lmp, name, dtype, count, data)
    check(lmp)
    return data
end

function pair_neighbor_list(lmp, name, exact, nsub, request)
    idx = API.lammps_find_pair_neighlist(lmp, name, exact, nsub, request)
    if idx == -1
        error("Could not find neighbor list for pair $(name)")
    end
    return idx
end

function fix_neighbor_list(lmp, name, request)
    idx = API.lammps_find_fix_neighlist(lmp, name, request)
    if idx == -1
        error("Could not find neighbor list for fix $(name)")
    end
    return idx
end


"""
    neighbors(lmb::LMP, idx, element)

Given a neighbor list `idx` and the element therein,
return the atom index, and it's neigbors.
"""
function neighbors(lmp, idx, element)
    r_iatom = Ref{Cint}()
    r_numneigh = Ref{Cint}()
    r_neighbors = Ref{Ptr{Cint}}(0)

    API.lammps_neighlist_element_neighbors(lmp, idx, element - 1, r_iatom, r_numneigh, r_neighbors)

    return r_iatom[], Base.unsafe_wrap(Array, r_neighbors[], r_numneigh[]; own = false)
end

include("external.jl")

end # module
