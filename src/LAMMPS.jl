module LAMMPS

include("api.jl")

export LMP

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

end # module
