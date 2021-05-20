using Clang.Generators

import LAMMPS_jll

header_dir = joinpath(LAMMPS_jll.artifact_dir, "include", "lammps")
isdir(header_dir) || error("$header_dir does not exist")

const LAMMPS_INCLUDE = normpath(header_dir)

args = ["-I$LAMMPS_INCLUDE"]

options = load_options(joinpath(@__DIR__, "wrap.toml"))

ctx = create_context(joinpath(LAMMPS_INCLUDE, "library.h"), args, options)
build!(ctx)

