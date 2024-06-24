# # SNAP example using GaN
#
# In this example we extract the bispectrum from SNAP.
# This example demonstrates how to install and use LAMMPS.jl
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed

# ```julia
# using Pkg
# pkg"add LAMMPS"
# ```

# LAMMPS.jl will automatically download and install a pre-built version of LAMMPS.
#
# We start off by importing `LAMMPS`.

using LAMMPS

const DATA = joinpath(dirname(pathof(LAMMPS)), "..", "examples", "example_GaN_data")

function run_snap(lmp, path, rcut, twojmax)
    command(lmp, """
        log none
        units metal
        boundary p p p
        atom_style atomic
        atom_modify map array
        read_data $path
        pair_style zero $rcut
        pair_coeff * *
        compute PE all pe
        compute S all pressure thermo_temp
        compute SNA all sna/atom $rcut 0.99363 $twojmax 0.5 0.5 1.0 0.5 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1
        compute SNAD all snad/atom $rcut 0.99363 $twojmax 0.5 0.5 1.0 0.5 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1
        compute SNAV all snav/atom $rcut 0.99363 $twojmax 0.5 0.5 1.0 0.5 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1
        thermo_style custom pe
        run 0
    """)

    ## Extract bispectrum
    bs = gather(lmp, "SNA", Float64)
    return bs
end

function calculate_snap_bispectrum(path, rcut, twojmax, M, N1, N2)
    J = twojmax / 2
    ncoeff = round(Int, (J+1)*(J+2)*((J+(1.5))/3) + 1)

    A = LMP(["-screen","none"]) do lmp
        A = Array{Float64}(undef, M, 2*ncoeff) # bispectrum is 2*(ncoeff - 1) + 2
        for m in 1:M
            data = joinpath(path, string(m), "DATA")
            bs = run_snap(lmp, data, rcut, twojmax)

            ## Make bispectrum sum vector
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
    return A, ncoeff
end

const rcut = 3.5
const twojmax = 6
const M  = 48 # number of input files
const N1 = 96 # number of atoms of the first type
const N2 = 96 # number of atoms of the second type
A, ncoeff = calculate_snap_bispectrum(DATA, rcut, twojmax, M, N1, N2)
