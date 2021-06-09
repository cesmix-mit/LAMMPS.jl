module LAMMPS

include("api.jl")

export LMP, command, get_natoms, extract_atom, extract_compute

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

function get_natoms(lmp::LMP)
    API.lammps_get_natoms(lmp)
end

function extract_atom(lmp::LMP, name, dtype::Union{Nothing, API._LMP_DATATYPE_CONST} = nothing)
    if dtype === nothing
        dtype = API.lammps_extract_atom_datatype(lmp, name)
    end
    dtype = convert(API._LMP_DATATYPE_CONST, dtype)

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
    else
        @assert false "Unknown dtype: $dtype"
    end
    ptr = API.lammps_extract_atom(lmp, name)
    reinterpret(type, ptr)
end

function extract_compute(lmp, name, style, type)
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

end # module
