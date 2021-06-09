using Test
using LAMMPS

let lmp = LAMMPS.LMP()
    @test LAMMPS.API.lammps_version(lmp) >= 0
end
