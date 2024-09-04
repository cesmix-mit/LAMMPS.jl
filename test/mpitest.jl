using Test
using LAMMPS
using MPI

MPI.Init()

LMP(["-screen", "none"], MPI.COMM_WORLD) do lmp
    extract_setting(lmp, "world_size") == 2 || MPI.Abort(MPI.COMM_WORLD, 1)
    command(lmp, "clear")
end

MPI.Finalize()