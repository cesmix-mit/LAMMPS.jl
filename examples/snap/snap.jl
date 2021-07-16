using Base: Float64
using LAMMPS

###
# Setup the matrices for least squares fitting
# Ax = b
# A - Bispectrum matrix
# x - unknowns coefficients
# b - energies, forces, stresses
###

const ntypes = 2
const twojmax = 5

const J = twojmax/2.
const ncoeff = round(Int, (J+1)*(J+2)*((J+(1.5))/3.) + 1)
@show ncoeff

A = LMP(["-screen","none"]) do lmp

    M = 48
    A = Array{Float64}(undef, M, 2*ncoeff) # bispectrum is 2*(ncoeff - 1) + 2

    for m in 1:M
        read_data_str = "read_data " * joinpath("data", string(m), "DATA")

        command(lmp, "units metal")
        command(lmp, "boundary p p p")
        command(lmp, "atom_style atomic")
        command(lmp, "atom_modify map array")
        command(lmp, read_data_str)
        command(lmp, "pair_style snap")
        command(lmp, "pair_coeff * * GaN.snapcoeff GaN.snapparam Ga N")
        command(lmp, "compute PE all pe")
        command(lmp, "compute S all pressure thermo_temp")
        command(lmp, "compute SNA all sna/atom 3.5 0.99363 6 0.5 0.5 1.0 0.5 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1")
        command(lmp, "compute SNAD all snad/atom 3.5 0.99363 6 0.5 0.5 1.0 0.5 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1")
        command(lmp, "compute SNAV all snav/atom 3.5 0.99363 6 0.5 0.5 1.0 0.5 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1")
        command(lmp, "thermo_style custom pe")
        command(lmp, "dump 2 all custom 100 dump.forces fx fy fz")
        command(lmp, "run 0")

        nlocal = extract_global(lmp, "nlocal")

        types = extract_atom(lmp, "type", LAMMPS.API.LAMMPS_INT)
        ids = extract_atom(lmp, "id", LAMMPS.API.LAMMPS_INT)

        @info "Progress" m nlocal

        ###
        # Energy
        ###

        bs = extract_compute(lmp, "SNA", LAMMPS.API.LMP_STYLE_ATOM,
                                         LAMMPS.API.LMP_TYPE_ARRAY)


        N1 = 96 # Number of type 1
        N2 = 96 # Number of type 2

        @assert count(==(1), types) == N1
        @assert count(==(2), types) == N2

        # Make bispectrum sum vector
        row = Float64[]

        push!(row, N1)

        for k in 1:(ncoeff-1)
            acc = 0.0
            for n in 1:N1
                acc += bs[k, n]
            end
            push!(row, acc)
        end

        push!(row, N2)
        for k in 1:(ncoeff-1)
            acc = 0.0
            for n in N1 .+ (1:N2)
                acc += bs[k, n]
            end
            push!(row, acc)
        end

        A[m, :] = row

        command(lmp, "clear")
    end
    return A
end
