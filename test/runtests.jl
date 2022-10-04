using Test
using LAMMPS
using MPI

LMP() do lmp
    @test LAMMPS.version(lmp) >= 0
end

LMP(["-screen", "none"]) do lmp
    @test LAMMPS.version(lmp) >= 0
    command(lmp, "clear")

    @test_throws ErrorException command(lmp, "nonsense")
end

MPI.mpiexec() do mpiexec
    @test success(pipeline(`$mpiexec -n 2 $(Base.julia_cmd()) mpitest.jl`, stderr=stderr, stdout=stdout))
end
