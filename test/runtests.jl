using Test
using LAMMPS

LMP() do lmp
    @test LAMMPS.version(lmp) >= 0
end

LMP(["-screen", "myscreen"]) do lmp
    @test LAMMPS.version(lmp) >= 0
end
