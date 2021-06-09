module LAMMPS

include("api.jl")

export LMP, command

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

end # module
