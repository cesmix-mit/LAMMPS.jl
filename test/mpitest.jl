using Test
using LAMMPS
using MPI

MPI.Init()

LMP(["-screen", "none"], MPI.COMM_WORLD) do lmp
    @test LAMMPS.version(lmp) >= 0
    command(lmp, "clear")
end