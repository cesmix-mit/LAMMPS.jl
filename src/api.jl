module API

using CEnum

import LAMMPS_jll
import LAMMPS_jll: liblammps
import MPI: MPI_Comm

@cenum _LMP_DATATYPE_CONST::UInt32 begin
    LAMMPS_INT = 0
    LAMMPS_INT_2D = 1
    LAMMPS_DOUBLE = 2
    LAMMPS_DOUBLE_2D = 3
    LAMMPS_INT64 = 4
    LAMMPS_INT64_2D = 5
    LAMMPS_STRING = 6
end

@cenum _LMP_STYLE_CONST::UInt32 begin
    LMP_STYLE_GLOBAL = 0
    LMP_STYLE_ATOM = 1
    LMP_STYLE_LOCAL = 2
end

@cenum _LMP_TYPE_CONST::UInt32 begin
    LMP_TYPE_SCALAR = 0
    LMP_TYPE_VECTOR = 1
    LMP_TYPE_ARRAY = 2
    LMP_SIZE_VECTOR = 3
    LMP_SIZE_ROWS = 4
    LMP_SIZE_COLS = 5
end

@cenum _LMP_ERROR_CONST::UInt32 begin
    LMP_ERROR_WARNING = 0
    LMP_ERROR_ONE = 1
    LMP_ERROR_ALL = 2
    LMP_ERROR_WORLD = 4
    LMP_ERROR_UNIVERSE = 8
end

@cenum _LMP_VAR_CONST::UInt32 begin
    LMP_VAR_EQUAL = 0
    LMP_VAR_ATOM = 1
    LMP_VAR_VECTOR = 2
    LMP_VAR_STRING = 3
end

function lammps_open(argc, argv, comm, ptr)
    ccall((:lammps_open, liblammps), Ptr{Cvoid}, (Cint, Ptr{Ptr{Cchar}}, MPI_Comm, Ptr{Ptr{Cvoid}}), argc, argv, comm, ptr)
end

function lammps_open_no_mpi(argc, argv, ptr)
    ccall((:lammps_open_no_mpi, liblammps), Ptr{Cvoid}, (Cint, Ptr{Ptr{Cchar}}, Ptr{Ptr{Cvoid}}), argc, argv, ptr)
end

function lammps_open_fortran(argc, argv, f_comm)
    ccall((:lammps_open_fortran, liblammps), Ptr{Cvoid}, (Cint, Ptr{Ptr{Cchar}}, Cint), argc, argv, f_comm)
end

function lammps_close(handle)
    ccall((:lammps_close, liblammps), Cvoid, (Ptr{Cvoid},), handle)
end

# no prototype is found for this function at library.h:128:6, please use with caution
function lammps_mpi_init()
    ccall((:lammps_mpi_init, liblammps), Cvoid, ())
end

# no prototype is found for this function at library.h:129:6, please use with caution
function lammps_mpi_finalize()
    ccall((:lammps_mpi_finalize, liblammps), Cvoid, ())
end

# no prototype is found for this function at library.h:130:6, please use with caution
function lammps_kokkos_finalize()
    ccall((:lammps_kokkos_finalize, liblammps), Cvoid, ())
end

# no prototype is found for this function at library.h:131:6, please use with caution
function lammps_python_finalize()
    ccall((:lammps_python_finalize, liblammps), Cvoid, ())
end

function lammps_error(handle, error_type, error_text)
    ccall((:lammps_error, liblammps), Cvoid, (Ptr{Cvoid}, Cint, Ptr{Cchar}), handle, error_type, error_text)
end

function lammps_file(handle, file)
    ccall((:lammps_file, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cchar}), handle, file)
end

function lammps_command(handle, cmd)
    ccall((:lammps_command, liblammps), Ptr{Cchar}, (Ptr{Cvoid}, Ptr{Cchar}), handle, cmd)
end

function lammps_commands_list(handle, ncmd, cmds)
    ccall((:lammps_commands_list, liblammps), Cvoid, (Ptr{Cvoid}, Cint, Ptr{Ptr{Cchar}}), handle, ncmd, cmds)
end

function lammps_commands_string(handle, str)
    ccall((:lammps_commands_string, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cchar}), handle, str)
end

function lammps_get_natoms(handle)
    ccall((:lammps_get_natoms, liblammps), Cdouble, (Ptr{Cvoid},), handle)
end

function lammps_get_thermo(handle, keyword)
    ccall((:lammps_get_thermo, liblammps), Cdouble, (Ptr{Cvoid}, Ptr{Cchar}), handle, keyword)
end

function lammps_extract_box(handle, boxlo, boxhi, xy, yz, xz, pflags, boxflag)
    ccall((:lammps_extract_box, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}), handle, boxlo, boxhi, xy, yz, xz, pflags, boxflag)
end

function lammps_reset_box(handle, boxlo, boxhi, xy, yz, xz)
    ccall((:lammps_reset_box, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Cdouble, Cdouble), handle, boxlo, boxhi, xy, yz, xz)
end

function lammps_memory_usage(handle, meminfo)
    ccall((:lammps_memory_usage, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cdouble}), handle, meminfo)
end

function lammps_get_mpi_comm(handle)
    ccall((:lammps_get_mpi_comm, liblammps), MPI_Comm, (Ptr{Cvoid},), handle)
end

function lammps_extract_setting(handle, keyword)
    ccall((:lammps_extract_setting, liblammps), Cint, (Ptr{Cvoid}, Ptr{Cchar}), handle, keyword)
end

function lammps_extract_global_datatype(handle, name)
    ccall((:lammps_extract_global_datatype, liblammps), Cint, (Ptr{Cvoid}, Ptr{Cchar}), handle, name)
end

function lammps_extract_global(handle, name)
    ccall((:lammps_extract_global, liblammps), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cchar}), handle, name)
end

function lammps_extract_atom_datatype(handle, name)
    ccall((:lammps_extract_atom_datatype, liblammps), Cint, (Ptr{Cvoid}, Ptr{Cchar}), handle, name)
end

function lammps_extract_atom(handle, name)
    ccall((:lammps_extract_atom, liblammps), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cchar}), handle, name)
end

function lammps_extract_compute(handle, arg2, arg3, arg4)
    ccall((:lammps_extract_compute, liblammps), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cchar}, Cint, Cint), handle, arg2, arg3, arg4)
end

function lammps_extract_fix(handle, arg2, arg3, arg4, arg5, arg6)
    ccall((:lammps_extract_fix, liblammps), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cchar}, Cint, Cint, Cint, Cint), handle, arg2, arg3, arg4, arg5, arg6)
end

function lammps_extract_variable(handle, arg2, arg3)
    ccall((:lammps_extract_variable, liblammps), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cchar}, Ptr{Cchar}), handle, arg2, arg3)
end

function lammps_extract_variable_datatype(handle, name)
    ccall((:lammps_extract_variable_datatype, liblammps), Cint, (Ptr{Cvoid}, Ptr{Cchar}), handle, name)
end

function lammps_set_variable(arg1, arg2, arg3)
    ccall((:lammps_set_variable, liblammps), Cint, (Ptr{Cvoid}, Ptr{Cchar}, Ptr{Cchar}), arg1, arg2, arg3)
end

function lammps_gather_atoms(handle, name, type, count, data)
    ccall((:lammps_gather_atoms, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cchar}, Cint, Cint, Ptr{Cvoid}), handle, name, type, count, data)
end

function lammps_gather_atoms_concat(handle, name, type, count, data)
    ccall((:lammps_gather_atoms_concat, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cchar}, Cint, Cint, Ptr{Cvoid}), handle, name, type, count, data)
end

function lammps_gather_atoms_subset(handle, name, type, count, ndata, ids, data)
    ccall((:lammps_gather_atoms_subset, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cchar}, Cint, Cint, Cint, Ptr{Cint}, Ptr{Cvoid}), handle, name, type, count, ndata, ids, data)
end

function lammps_scatter_atoms(handle, name, type, count, data)
    ccall((:lammps_scatter_atoms, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cchar}, Cint, Cint, Ptr{Cvoid}), handle, name, type, count, data)
end

function lammps_scatter_atoms_subset(handle, name, type, count, ndata, ids, data)
    ccall((:lammps_scatter_atoms_subset, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cchar}, Cint, Cint, Cint, Ptr{Cint}, Ptr{Cvoid}), handle, name, type, count, ndata, ids, data)
end

function lammps_gather_bonds(handle, data)
    ccall((:lammps_gather_bonds, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), handle, data)
end

function lammps_gather_angles(handle, data)
    ccall((:lammps_gather_angles, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), handle, data)
end

function lammps_gather_dihedrals(handle, data)
    ccall((:lammps_gather_dihedrals, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), handle, data)
end

function lammps_gather_impropers(handle, data)
    ccall((:lammps_gather_impropers, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), handle, data)
end

function lammps_gather(handle, name, type, count, data)
    ccall((:lammps_gather, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cchar}, Cint, Cint, Ptr{Cvoid}), handle, name, type, count, data)
end

function lammps_gather_concat(handle, name, type, count, data)
    ccall((:lammps_gather_concat, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cchar}, Cint, Cint, Ptr{Cvoid}), handle, name, type, count, data)
end

function lammps_gather_subset(handle, name, type, count, ndata, ids, data)
    ccall((:lammps_gather_subset, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cchar}, Cint, Cint, Cint, Ptr{Cint}, Ptr{Cvoid}), handle, name, type, count, ndata, ids, data)
end

function lammps_scatter(handle, name, type, count, data)
    ccall((:lammps_scatter, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cchar}, Cint, Cint, Ptr{Cvoid}), handle, name, type, count, data)
end

function lammps_scatter_subset(handle, name, type, count, ndata, ids, data)
    ccall((:lammps_scatter_subset, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cchar}, Cint, Cint, Cint, Ptr{Cint}, Ptr{Cvoid}), handle, name, type, count, ndata, ids, data)
end

function lammps_create_atoms(handle, n, id, type, x, v, image, bexpand)
    ccall((:lammps_create_atoms, liblammps), Cint, (Ptr{Cvoid}, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Cint), handle, n, id, type, x, v, image, bexpand)
end

function lammps_find_pair_neighlist(handle, style, exact, nsub, request)
    ccall((:lammps_find_pair_neighlist, liblammps), Cint, (Ptr{Cvoid}, Ptr{Cchar}, Cint, Cint, Cint), handle, style, exact, nsub, request)
end

function lammps_find_fix_neighlist(handle, id, request)
    ccall((:lammps_find_fix_neighlist, liblammps), Cint, (Ptr{Cvoid}, Ptr{Cchar}, Cint), handle, id, request)
end

function lammps_find_compute_neighlist(handle, id, request)
    ccall((:lammps_find_compute_neighlist, liblammps), Cint, (Ptr{Cvoid}, Ptr{Cchar}, Cint), handle, id, request)
end

function lammps_neighlist_num_elements(handle, idx)
    ccall((:lammps_neighlist_num_elements, liblammps), Cint, (Ptr{Cvoid}, Cint), handle, idx)
end

function lammps_neighlist_element_neighbors(handle, idx, element, iatom, numneigh, neighbors)
    ccall((:lammps_neighlist_element_neighbors, liblammps), Cvoid, (Ptr{Cvoid}, Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Ptr{Cint}}), handle, idx, element, iatom, numneigh, neighbors)
end

function lammps_version(handle)
    ccall((:lammps_version, liblammps), Cint, (Ptr{Cvoid},), handle)
end

function lammps_get_os_info(buffer, buf_size)
    ccall((:lammps_get_os_info, liblammps), Cvoid, (Ptr{Cchar}, Cint), buffer, buf_size)
end

# no prototype is found for this function at library.h:231:5, please use with caution
function lammps_config_has_mpi_support()
    ccall((:lammps_config_has_mpi_support, liblammps), Cint, ())
end

# no prototype is found for this function at library.h:232:5, please use with caution
function lammps_config_has_gzip_support()
    ccall((:lammps_config_has_gzip_support, liblammps), Cint, ())
end

# no prototype is found for this function at library.h:233:5, please use with caution
function lammps_config_has_png_support()
    ccall((:lammps_config_has_png_support, liblammps), Cint, ())
end

# no prototype is found for this function at library.h:234:5, please use with caution
function lammps_config_has_jpeg_support()
    ccall((:lammps_config_has_jpeg_support, liblammps), Cint, ())
end

# no prototype is found for this function at library.h:235:5, please use with caution
function lammps_config_has_ffmpeg_support()
    ccall((:lammps_config_has_ffmpeg_support, liblammps), Cint, ())
end

# no prototype is found for this function at library.h:236:5, please use with caution
function lammps_config_has_exceptions()
    ccall((:lammps_config_has_exceptions, liblammps), Cint, ())
end

function lammps_config_has_package(arg1)
    ccall((:lammps_config_has_package, liblammps), Cint, (Ptr{Cchar},), arg1)
end

# no prototype is found for this function at library.h:239:5, please use with caution
function lammps_config_package_count()
    ccall((:lammps_config_package_count, liblammps), Cint, ())
end

function lammps_config_package_name(arg1, arg2, arg3)
    ccall((:lammps_config_package_name, liblammps), Cint, (Cint, Ptr{Cchar}, Cint), arg1, arg2, arg3)
end

function lammps_config_accelerator(arg1, arg2, arg3)
    ccall((:lammps_config_accelerator, liblammps), Cint, (Ptr{Cchar}, Ptr{Cchar}, Ptr{Cchar}), arg1, arg2, arg3)
end

# no prototype is found for this function at library.h:243:5, please use with caution
function lammps_has_gpu_device()
    ccall((:lammps_has_gpu_device, liblammps), Cint, ())
end

function lammps_get_gpu_device_info(buffer, buf_size)
    ccall((:lammps_get_gpu_device_info, liblammps), Cvoid, (Ptr{Cchar}, Cint), buffer, buf_size)
end

function lammps_has_style(arg1, arg2, arg3)
    ccall((:lammps_has_style, liblammps), Cint, (Ptr{Cvoid}, Ptr{Cchar}, Ptr{Cchar}), arg1, arg2, arg3)
end

function lammps_style_count(arg1, arg2)
    ccall((:lammps_style_count, liblammps), Cint, (Ptr{Cvoid}, Ptr{Cchar}), arg1, arg2)
end

function lammps_style_name(arg1, arg2, arg3, arg4, arg5)
    ccall((:lammps_style_name, liblammps), Cint, (Ptr{Cvoid}, Ptr{Cchar}, Cint, Ptr{Cchar}, Cint), arg1, arg2, arg3, arg4, arg5)
end

function lammps_has_id(arg1, arg2, arg3)
    ccall((:lammps_has_id, liblammps), Cint, (Ptr{Cvoid}, Ptr{Cchar}, Ptr{Cchar}), arg1, arg2, arg3)
end

function lammps_id_count(arg1, arg2)
    ccall((:lammps_id_count, liblammps), Cint, (Ptr{Cvoid}, Ptr{Cchar}), arg1, arg2)
end

function lammps_id_name(arg1, arg2, arg3, arg4, arg5)
    ccall((:lammps_id_name, liblammps), Cint, (Ptr{Cvoid}, Ptr{Cchar}, Cint, Ptr{Cchar}, Cint), arg1, arg2, arg3, arg4, arg5)
end

# no prototype is found for this function at library.h:254:5, please use with caution
function lammps_plugin_count()
    ccall((:lammps_plugin_count, liblammps), Cint, ())
end

function lammps_plugin_name(arg1, arg2, arg3, arg4)
    ccall((:lammps_plugin_name, liblammps), Cint, (Cint, Ptr{Cchar}, Ptr{Cchar}, Cint), arg1, arg2, arg3, arg4)
end

function lammps_encode_image_flags(ix, iy, iz)
    ccall((:lammps_encode_image_flags, liblammps), Cint, (Cint, Cint, Cint), ix, iy, iz)
end

function lammps_decode_image_flags(image, flags)
    ccall((:lammps_decode_image_flags, liblammps), Cvoid, (Cint, Ptr{Cint}), image, flags)
end

# typedef void ( * FixExternalFnPtr ) ( void * , int64_t , int , int * , double * * , double * * )
const FixExternalFnPtr = Ptr{Cvoid}

function lammps_set_fix_external_callback(handle, id, funcptr, ptr)
    ccall((:lammps_set_fix_external_callback, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cchar}, FixExternalFnPtr, Ptr{Cvoid}), handle, id, funcptr, ptr)
end

function lammps_fix_external_get_force(handle, id)
    ccall((:lammps_fix_external_get_force, liblammps), Ptr{Ptr{Cdouble}}, (Ptr{Cvoid}, Ptr{Cchar}), handle, id)
end

function lammps_fix_external_set_energy_global(handle, id, eng)
    ccall((:lammps_fix_external_set_energy_global, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cchar}, Cdouble), handle, id, eng)
end

function lammps_fix_external_set_energy_peratom(handle, id, eng)
    ccall((:lammps_fix_external_set_energy_peratom, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cchar}, Ptr{Cdouble}), handle, id, eng)
end

function lammps_fix_external_set_virial_global(handle, id, virial)
    ccall((:lammps_fix_external_set_virial_global, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cchar}, Ptr{Cdouble}), handle, id, virial)
end

function lammps_fix_external_set_virial_peratom(handle, id, virial)
    ccall((:lammps_fix_external_set_virial_peratom, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cchar}, Ptr{Ptr{Cdouble}}), handle, id, virial)
end

function lammps_fix_external_set_vector_length(handle, id, len)
    ccall((:lammps_fix_external_set_vector_length, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cchar}, Cint), handle, id, len)
end

function lammps_fix_external_set_vector(handle, id, idx, val)
    ccall((:lammps_fix_external_set_vector, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cchar}, Cint, Cdouble), handle, id, idx, val)
end

function lammps_flush_buffers(ptr)
    ccall((:lammps_flush_buffers, liblammps), Cvoid, (Ptr{Cvoid},), ptr)
end

function lammps_fix_external_set_energy_peratom(handle, id, eng)
    ccall((:lammps_fix_external_set_energy_peratom, liblammps), Cvoid, (Ptr{Cvoid}, Ptr{Cchar}, Ptr{Cdouble}), handle, id, eng)
end

function lammps_free(ptr)
    ccall((:lammps_free, liblammps), Cvoid, (Ptr{Cvoid},), ptr)
end

function lammps_is_running(handle)
    ccall((:lammps_is_running, liblammps), Cint, (Ptr{Cvoid},), handle)
end

function lammps_force_timeout(handle)
    ccall((:lammps_force_timeout, liblammps), Cvoid, (Ptr{Cvoid},), handle)
end

function lammps_has_error(handle)
    ccall((:lammps_has_error, liblammps), Cint, (Ptr{Cvoid},), handle)
end

function lammps_get_last_error_message(handle, buffer, buf_size)
    ccall((:lammps_get_last_error_message, liblammps), Cint, (Ptr{Cvoid}, Ptr{Cchar}, Cint), handle, buffer, buf_size)
end

# no prototype is found for this function at library.h:297:5, please use with caution
function lammps_python_api_version()
    ccall((:lammps_python_api_version, liblammps), Cint, ())
end

end # module
