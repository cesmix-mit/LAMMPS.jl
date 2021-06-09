module LAMMPS

include("api.jl")

mutable struct LMP
    handle::Ptr{Cvoid}

    function LMP()
        handle = API.lammps_open_no_mpi(0, C_NULL, C_NULL)

        this = new(handle)
        finalizer(this) do this
            API.lammps_close(this)
        end
        return this
    end
end
Base.unsafe_convert(::Type{Ptr{Cvoid}}, lmp::LMP) = lmp.handle

function version(lmp::LMP)
    API.lammps_version(lmp)
end

end # module
