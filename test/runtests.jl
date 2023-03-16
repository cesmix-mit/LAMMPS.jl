using Test
using LAMMPS
using MPI

@test_logs (:warn,"LAMMPS library path changed, you will need to restart Julia for the change to take effect") LAMMPS.set_library!(LAMMPS.locate())

LMP() do lmp
    @test LAMMPS.version(lmp) >= 0
end

LMP(["-screen", "none"]) do lmp
    @test LAMMPS.version(lmp) >= 0
    command(lmp, "clear")

    @test_throws ErrorException command(lmp, "nonsense")
end


function f()
    lmp = LMP(["-screen", "none"])
    @test LAMMPS.version(lmp) >= 0
    command(lmp, "clear")
    @test_throws ErrorException command(lmp, "nonsense")
    LAMMPS.close!(lmp)
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

@testset "gather/scatter" begin
    LMP(["-screen", "none"]) do lmp
        # setting up example data
        command(lmp, "atom_modify map yes")
        command(lmp, "region cell block 0 3 0 3 0 3")
        command(lmp, "create_box 1 cell")
        command(lmp, "lattice sc 1")
        command(lmp, "create_atoms 1 region cell")
        command(lmp, "mass 1 1")

        command(lmp, "compute pos all property/atom x y z")
        command(lmp, "fix pos all ave/atom 10 1 10 c_pos[1] c_pos[2] c_pos[3]")

        command(lmp, "run 10")
        data = zeros(Float64, 3, 27)
        subset = Int32.([2,5,10, 5])
        data_subset = ones(Float64, 3, 4)

        subset_bad1 = Int32.([28])
        subset_bad2 = Int32.([0])
        subset_bad_data = ones(Float64, 3,1)

        @test_throws AssertionError gather(lmp, "x", Int32)
        @test_throws AssertionError gather(lmp, "id", Float64)

        @test_throws ErrorException gather(lmp, "nonesense", Float64)
        @test_throws ErrorException gather(lmp, "c_nonsense", Float64)
        @test_throws ErrorException gather(lmp, "f_nonesense", Float64)

        @test_throws AssertionError gather(lmp, "x", Float64, subset_bad1)
        @test_throws AssertionError gather(lmp, "x", Float64, subset_bad2)

        @test_throws ErrorException scatter!(lmp, "nonesense", data)
        @test_throws ErrorException scatter!(lmp, "c_nonsense", data)
        @test_throws ErrorException scatter!(lmp, "f_nonesense", data)

        @test_throws AssertionError scatter!(lmp, "x", subset_bad_data, subset_bad1)
        @test_throws AssertionError scatter!(lmp, "x", subset_bad_data, subset_bad2)

        @test gather(lmp, "x", Float64) == gather(lmp, "c_pos", Float64) == gather(lmp, "f_pos", Float64)

        @test gather(lmp, "x", Float64)[:,subset] == gather(lmp, "x", Float64, subset)
        @test gather(lmp, "c_pos", Float64)[:,subset] == gather(lmp, "c_pos", Float64, subset)
        @test gather(lmp, "f_pos", Float64)[:,subset] == gather(lmp, "f_pos", Float64, subset)

        scatter!(lmp, "x", data)
        scatter!(lmp, "f_pos", data)
        scatter!(lmp, "c_pos", data)

        @test gather(lmp, "x", Float64) == gather(lmp, "c_pos", Float64) == gather(lmp, "f_pos", Float64) == data

        scatter!(lmp, "x", data_subset, subset)
        scatter!(lmp, "c_pos", data_subset, subset)
        scatter!(lmp, "f_pos", data_subset, subset)

        @test gather(lmp, "x", Float64, subset) == gather(lmp, "c_pos", Float64, subset) == gather(lmp, "f_pos", Float64, subset) == data_subset

    end
end

LMP(["-screen", "none"]) do lmp
    called = Ref(false)
    command(lmp, "boundary p p p")
    command(lmp, "region cell block 0 1 0 1 0 1 units box")
    command(lmp, "create_box 1 cell")
    command(lmp, "fix julia all external pf/callback 1 1")
    LAMMPS.FixExternal(lmp, "julia") do fix, timestep, nlocal, ids, x, fexternal
       LAMMPS.energy_global!(fix, 0.0)
       called[] = true
    end
    command(lmp, "mass 1 1.0")
    command(lmp, "run 0")
    @test called[] == true
end

@test success(pipeline(`$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) mpitest.jl`, stderr=stderr, stdout=stdout))

include("external_pair.jl")
