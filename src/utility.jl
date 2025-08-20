"""
    eval(lmp::LMP, expr::String)

Evaluate an immediate variable expression

This function takes a string with an expression that can be used for equal style variables, evaluates it and returns the resulting (scalar) value as a floating point number.
"""
function eval(lmp::LMP, expr::String)
    result = API.lammps_eval(lmp, expr)
    check(lmp)
    result
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