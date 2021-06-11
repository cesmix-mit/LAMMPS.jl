module LAMMPS

include("api.jl")

export LMP, command, get_natoms, extract_atom, extract_compute, extract_global

mutable struct LMP
    handle::Ptr{Cvoid}

    function LMP(args::Vector{String}=String[])
        if isempty(args)
            argsv = C_NULL
        else
            args = copy(args)
            pushfirst!(args, "lammps")
            argsv = map(pointer, args)
        end

        GC.@preserve args begin
            handle = API.lammps_open_no_mpi(length(args), argsv, C_NULL)
        end

        this = new(handle)
        finalizer(this) do this
            API.lammps_close(this)
        end
        return this
    end
end
Base.unsafe_convert(::Type{Ptr{Cvoid}}, lmp::LMP) = lmp.handle

function LMP(f::Function, args=String[])
    lmp = LMP(args)
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

function extract_global(lmp::LMP, name, dtype=nothing)
    if dtype === nothing
        dtype = API.lammps_extract_global_datatype(lmp, name)
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
    if length(shape) == 2
        array = transpose(array)
    end
    return array
end

"""
    extract_atom(lmp, name, dtype=nothing, )
"""
function extract_atom(lmp::LMP, name,
                      dtype::Union{Nothing, API._LMP_DATATYPE_CONST} = nothing,
                      axes1=nothing, axes2=nothing)


    if dtype === nothing
        dtype = API.lammps_extract_atom_datatype(lmp, name)
    end
    dtype = convert(API._LMP_DATATYPE_CONST, dtype)

    if axes1 === nothing
        if name == "mass"
            axes1 = extract_global(lmp, "ntypes") + 1
        else
            axes1 = extract_global(lmp, "nlocal")
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

function unsafe_extract_compute(lmp, name, style, type)
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

    ptr = reinterpret(dtype, ptr)

    if extract
        return Base.unsafe_load(ptr)
    end
    return ptr
end

function extract_compute(lmp, name, style, type)
    ptr = unsafe_extract_compute(lmp, name, style, type)

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

end # module
