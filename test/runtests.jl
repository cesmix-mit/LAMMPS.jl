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

@testset "Variables" begin
    LMP(["-screen", "none"]) do lmp
        command(lmp, "box tilt large")
        command(lmp, "region cell block 0 1.0 0 1.0 0 1.0 units box")
        command(lmp, "create_box 1 cell")
        command(lmp, "create_atoms 1 random 10 1 NULL")
        command(lmp, "compute  press all pressure NULL pair");
        command(lmp, "fix press all ave/time 1 1 1 c_press mode vector");

        command(lmp, "variable var1 equal 1.0")
        command(lmp, "variable var2 string \"hello\"")
        command(lmp, "variable var3 atom x")
        # TODO: x is 3d, how do we access more than the first dims
        command(lmp, "variable var4 vector f_press")

        @test LAMMPS.extract_variable(lmp, "var1") == 1.0
        @test LAMMPS.extract_variable(lmp, "var2") == "hello"
        x = LAMMPS.extract_atom(lmp, "x")
        x_var = LAMMPS.extract_variable(lmp, "var3")
        @test length(x_var) == 10
        @test x_var == x[1, :]
        press = LAMMPS.extract_variable(lmp, "var4")
        @test press isa Vector{Float64}
    end
end

MPI.mpiexec() do mpiexec
    @test success(pipeline(`$mpiexec -n 2 $(Base.julia_cmd()) mpitest.jl`, stderr=stderr, stdout=stdout))
end
