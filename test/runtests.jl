using Test
using LAMMPS

LAMMPS.LMP() do lmp
    @test LAMMPS.API.lammps_version(lmp) >= 0
end
