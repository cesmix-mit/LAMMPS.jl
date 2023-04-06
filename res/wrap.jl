using Clang.Generators

import LAMMPS_jll
# import MPItrampoline_jll

header_dir = joinpath(LAMMPS_jll.artifact_dir, "include", "lammps")
isdir(header_dir) || error("$header_dir does not exist")

# mpi_header_dir = joinpath(MPItrampoline_jll.artifact_dir, "include")
# isdir(mpi_header_dir) || error("$mpi_header_dir does not exist")

const LAMMPS_INCLUDE = normpath(header_dir)
# const MPI_INCLUDE = normpath(mpi_header_dir)

args = get_default_args()
push!(args, "-I$LAMMPS_INCLUDE")

push!(args, "-DLAMMPS_LIB_MPI")

options = load_options(joinpath(@__DIR__, "wrap.toml"))

ctx = create_context(joinpath(LAMMPS_INCLUDE, "library.h"), args, options)
build!(ctx)

