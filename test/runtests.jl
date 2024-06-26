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

@testset "Variables" begin
    LMP(["-screen", "none"]) do lmp
        command(lmp, """
            box tilt large
            region cell block 0 1.0 0 1.0 0 1.0 units box
            create_box 1 cell
            create_atoms 1 random 10 1 NULL
            compute  press all pressure NULL pair
            fix press all ave/time 1 1 1 c_press mode vector

            variable var1 equal 1.0
            variable var2 string \"hello\"
            variable var3 atom x
            # TODO: x is 3d, how do we access more than the first dims
            variable var4 vector f_press
        """)

        @test extract_variable(lmp, "var1", VARIABLE_EQUAL) == 1.0
        @test extract_variable(lmp, "var2", VARIABLE_STRING) == "hello"
        x = extract_atom(lmp, "x", LAMMPS_DOUBLE_2D)
        x_var = extract_variable(lmp, "var3", VARIABLE_ATOM)
        @test length(x_var) == 10
        @test x_var == x[1, :]
        press = LAMMPS.extract_variable(lmp, "var4", VARIABLE_VECTOR)
        @test press isa Vector{Float64}
    end
end

@testset "gather/scatter" begin
    LMP(["-screen", "none"]) do lmp
        # setting up example data
        command(lmp, """
            atom_modify map yes
            region cell block 0 3 0 3 0 3
            create_box 1 cell
            lattice sc 1
            create_atoms 1 region cell
            mass 1 1

            compute pos all property/atom x y z
            fix pos all ave/atom 10 1 10 c_pos[*]

            run 10
        """)
        
        data = zeros(Float64, 3, 27)
        subset = Int32.([2,5,10, 5])
        data_subset = ones(Float64, 3, 4)

        subset_bad1 = Int32.([28])
        subset_bad2 = Int32.([0])
        subset_bad_data = ones(Float64, 3,1)

        @test_throws AssertionError gather(lmp, "x", LAMMPS_INT_2D)
        @test_throws AssertionError gather(lmp, "id", LAMMPS_DOUBLE)

        @test_throws ErrorException gather(lmp, "nonesense", LAMMPS_DOUBLE_2D)
        @test_throws ErrorException gather(lmp, "c_nonsense", LAMMPS_DOUBLE_2D)
        @test_throws ErrorException gather(lmp, "f_nonesense", LAMMPS_DOUBLE_2D)

        @test_throws AssertionError gather(lmp, "x", LAMMPS_DOUBLE_2D, subset_bad1)
        @test_throws AssertionError gather(lmp, "x", LAMMPS_DOUBLE_2D, subset_bad2)

        @test_throws ErrorException scatter!(lmp, "nonesense", data)
        @test_throws ErrorException scatter!(lmp, "c_nonsense", data)
        @test_throws ErrorException scatter!(lmp, "f_nonesense", data)

        @test_throws AssertionError scatter!(lmp, "x", subset_bad_data, subset_bad1)
        @test_throws AssertionError scatter!(lmp, "x", subset_bad_data, subset_bad2)

        @test gather(lmp, "x", LAMMPS_DOUBLE_2D) == gather(lmp, "c_pos", LAMMPS_DOUBLE_2D) == gather(lmp, "f_pos", LAMMPS_DOUBLE_2D)

        @test gather(lmp, "x", LAMMPS_DOUBLE_2D)[:,subset] == gather(lmp, "x", LAMMPS_DOUBLE_2D, subset)
        @test gather(lmp, "c_pos", LAMMPS_DOUBLE_2D)[:,subset] == gather(lmp, "c_pos", LAMMPS_DOUBLE_2D, subset)
        @test gather(lmp, "f_pos", LAMMPS_DOUBLE_2D)[:,subset] == gather(lmp, "f_pos", LAMMPS_DOUBLE_2D, subset)

        scatter!(lmp, "x", data)
        scatter!(lmp, "f_pos", data)
        scatter!(lmp, "c_pos", data)

        @test gather(lmp, "x", LAMMPS_DOUBLE_2D) == gather(lmp, "c_pos", LAMMPS_DOUBLE_2D) == gather(lmp, "f_pos", LAMMPS_DOUBLE_2D) == data

        scatter!(lmp, "x", data_subset, subset)
        scatter!(lmp, "c_pos", data_subset, subset)
        scatter!(lmp, "f_pos", data_subset, subset)

        @test gather(lmp, "x", LAMMPS_DOUBLE_2D, subset) == gather(lmp, "c_pos", LAMMPS_DOUBLE_2D, subset) == gather(lmp, "f_pos", LAMMPS_DOUBLE_2D, subset) == data_subset

    end
end

@testset "Utilities" begin
    LMP(["-screen", "none"]) do lmp
        # setting up example data
        command(lmp, """
            atom_modify map yes
            region cell block 0 2 0 2 0 2
            create_box 1 cell
            lattice sc 1
            create_atoms 1 region cell
            mass 1 1

            group a id 1 2 3 5 8
            group even id 2 4 6 8
            group odd id 1 3 5 7
        """)

        @test group_to_atom_ids(lmp, "all") == 1:8
        @test group_to_atom_ids(lmp, "a") == [1, 2, 3, 5, 8]
        @test group_to_atom_ids(lmp, "even") == [2, 4, 6, 8]
        @test group_to_atom_ids(lmp, "odd") == [1, 3, 5, 7]
        @test_throws ErrorException group_to_atom_ids(lmp, "nonesense")

        command(lmp, [
            "compute pos all property/atom x y z",
            "fix 1 all ave/atom 10 1 10 c_pos[*]",
            "run 10"
        ])

        @test get_category_ids(lmp, "group") == ["all", "a", "even", "odd"]
        @test get_category_ids(lmp, "compute") == ["thermo_temp", "thermo_press", "thermo_pe", "pos"] # some of these computes are there by default it seems
        @test get_category_ids(lmp, "fix") == ["1"]
        @test_throws ErrorException get_category_ids(lmp, "nonesense")
    end
end

@test success(pipeline(`$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) mpitest.jl`, stderr=stderr, stdout=stdout))
