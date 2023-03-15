function fix_external_callback end

mutable struct FixExternal
    lmp::LMP
    name::String
    callback

    function FixExternal(lmp::LMP, name::String, callback)
        if haskey(lmp.external_fixes, name)
            error("FixExternal has already been registered with $name")
        end

        this = new(lmp, name, callback)
        lmp.external_fixes[name] = this # preserves pair globally

        ctx = Base.pointer_from_objref(this)
        callback = @cfunction(fix_external_callback, Cvoid, (Ptr{Cvoid}, Int64, Cint, Ptr{Cint}, Ptr{Ptr{Float64}}, Ptr{Ptr{Float64}}))
        API.lammps_set_fix_external_callback(lmp, name, callback, ctx)

        return this
    end
end

FixExternal(callback, lmp::LMP, name::String) = FixExternal(lmp::LMP, name::String, callback)

function fix_external_callback(ctx::Ptr{Cvoid}, timestep::Int64, nlocal::Cint, ids::Ptr{Cint}, x::Ptr{Ptr{Float64}}, fexternal::Ptr{Ptr{Float64}})
    fix = Base.unsafe_pointer_to_objref(ctx)::FixExternal

    @debug "Calling fix_external_callback on" fix timestep ids x fexternal
    # TODO wrap as arrays
    fix.callback(fix, timestep, nlocal, ids, x, fexternal)
    return nothing
end

function energy_global!(fix::FixExternal, energy)
    API.lammps_fix_external_set_energy_global(fix.lmp, fix.name, energy)
end

# TODO
# virial_global!
# function virial_global!(fix::FixExternal, )

