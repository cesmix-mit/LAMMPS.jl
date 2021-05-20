using Test
using LAMMPS

let lmp = LAMMPS.Session()
    @test LAMMPS.API.lammps_version(lmp) >= 0
end