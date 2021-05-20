module LAMMPS

include("api.jl")

mutable struct Session
    handle::Ptr{Cvoid}

    function Session()
        handle = API.lammps_open_no_mpi(0, C_NULL, C_NULL)

        this = new(handle)
        finalizer(this) do this
            API.lammps_close(this)
        end
        return this
    end
end
Base.unsafe_convert(::Type{Ptr{Cvoid}}, lmp::Session) = lmp.handle

function version(lmp::Session)
    API.lammps_version(lmp)
end

end # module
