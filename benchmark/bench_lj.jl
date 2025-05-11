module BenchmarkLJ

using BenchmarkTools
using MPI; MPI.Init()
using LAMMPS
import Enzyme, ForwardDiff
import ADTypes: AutoEnzyme, AutoForwardDiff

function setup_reference()
    lmp = LMP(["-screen", "none", "-log", "none"])

    command(lmp, raw"""
        # 3d Lennard-Jones melt

        variable        x index 1
        variable        y index 1
        variable        z index 1

        variable        xx equal 20*$x
        variable        yy equal 20*$y
        variable        zz equal 20*$z

        units           lj
        atom_style      atomic

        lattice         fcc 0.8442
        region          box block 0 ${xx} 0 ${yy} 0 ${zz}
        create_box      1 box
        create_atoms    1 box
        mass            1 1.0

        velocity        all create 1.44 87287 loop geom

        pair_style      lj/cut 2.5
        pair_coeff      1 1 1.0 1.0 2.5

        neighbor        0.3 bin
        neigh_modify    delay 0 every 20 check no

        fix             1 all nve
    """)

    return lmp
end

function setup_external(backend)
    lmp = LMP(["-screen", "none", "-log", "none"])

    command(lmp, raw"""
        variable        x index 1
        variable        y index 1
        variable        z index 1

        variable        xx equal 20*$x
        variable        yy equal 20*$y
        variable        zz equal 20*$z

        units           lj
        atom_style      atomic

        lattice         fcc 0.8442
        region          box block 0 ${xx} 0 ${yy} 0 ${zz}
        create_box      1 box
        create_atoms    1 box
        mass            1 1.0

        velocity        all create 1.44 87287 loop geom

        neighbor        0.3 bin
        neigh_modify    delay 0 every 20 check no
    """)

    config = InteractionConfig(; backend)
    coeff = [(1.0, 1.0)]

    if backend === nothing
        PairExternal(lmp, config, 2.5) do r, system, iatom, jatom
            @inbounds ε, σ = coeff[iatom.type, jatom.type]
            r6inv = (σ/r)^6
            energy = 4ε * (r6inv * (r6inv - 1))
            force = 24ε * (r6inv * (2r6inv - 1)) / r
            return energy, force    
        end
    else
        PairExternal(lmp, config, 2.5) do r, system, iatom, jatom
            @inbounds ε, σ = coeff[iatom.type, jatom.type]
            r6inv = (σ/r)^6
            return 4ε * (r6inv * (r6inv - 1))
        end
    end

    command(lmp, "fix 1 all nve")
    return lmp
end

suite = BenchmarkGroup()

suite["Reference"] = @benchmarkable command(lmp, "run 100") setup = (lmp = setup_reference())
suite["ForwardDiff"] = @benchmarkable command(lmp, "run 100") setup = (lmp = setup_external(AutoForwardDiff()))
suite["Enzyme"] = @benchmarkable command(lmp, "run 100") setup = (lmp = setup_external(AutoEnzyme()))
suite["Manual"] = @benchmarkable command(lmp, "run 100") setup = (lmp = setup_external(nothing))

end
BenchmarkLJ.suite