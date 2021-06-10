using Test
using LAMMPS

LMP() do lmp
    @test LAMMPS.version(lmp) >= 0
end

LMP(["-screen", "none"]) do lmp
    @test LAMMPS.version(lmp) >= 0
    command(lmp, "clear")

    @test_throws ErrorException command(lmp, "nonsense")
end

include("examples.jl")
