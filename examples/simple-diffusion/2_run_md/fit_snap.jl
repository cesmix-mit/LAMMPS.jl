# # SNAP example for hydrogen diffusing in palladium. 

using LAMMPS
using Printf

const DATA = joinpath(dirname(pathof(LAMMPS)), "..", "examples", "simple-diffusion", "2_run_md", "data")
const ntypes = 2
const rcut = 2.0
const twojmax = 6
const M  = 100 # number of input files
const N1 = 1 # number of atoms of the first type
const N2 = 64 # number of atoms of the second type
# SNAP radii and weights (see documentation on compute sna/atom).
const r1 = 0.5
const r2 = 0.5
const w1 = 0.5
const w2 = 1.0

if (twojmax%2==0)
    mm = (twojmax/2) + 1
    K = mm*(mm+1)*(2*mm+1)/6
end

@printf("Number of bispectrum components per atom: %d\n", K)

J = twojmax / 2
ncoeff = round(Int, (J+1)*(J+2)*((J+(1.5))/3) + 1)
@printf("Number of bispectrum coefficients: %d\n", ncoeff)


function calc_bispectrum(lmp, path, rcut, twojmax)
    read_data_str = "read_data " * path

    #print(read_data_str)

    command(lmp, "log none")
    command(lmp, "units metal")
    command(lmp, "boundary p p p")
    command(lmp, "atom_style atomic")
    command(lmp, "atom_modify map array")
    command(lmp, read_data_str)
    command(lmp, "pair_style zero $rcut")
    command(lmp, "pair_coeff * *")
    command(lmp, "compute PE all pe")
    command(lmp, "compute S all pressure thermo_temp")
    #command(lmp, "compute SNAP all snap $rcut 0.99363 $twojmax 0.5 0.5 1.0 0.5 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1")
    command(lmp, "compute SNA all sna/atom $rcut 0.99363 $twojmax $r1 $r2 $w1 $w2 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1")
    command(lmp, "compute SNAD all snad/atom $rcut 0.99363 $twojmax $r1 $r2 $w1 $w2 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1")
    command(lmp, "compute SNAV all snav/atom $rcut 0.99363 $twojmax $r1 $r2 $w1 $w2 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1")
    command(lmp, "thermo_style custom pe")
    command(lmp, "run 0")

    ## Extract bispectrum
    bs = extract_compute(lmp, "SNA", LAMMPS.API.LMP_STYLE_ATOM,
                                     LAMMPS.API.LMP_TYPE_ARRAY)

    bsd = extract_compute(lmp, "SNAD", LAMMPS.API.LMP_STYLE_ATOM, LAMMPS.API.LMP_TYPE_ARRAY)

    #btest = extract_compute(lmp, "SNAP", LAMMPS.API.LMP_STYLE_GLOBAL, LAMMPS.API.LMP_TYPE_ARRAY)
    #sz_btest = size(btest)
    #print(sz_btest)
    #print("\n")

    types = extract_atom(lmp, "type")
    #print(types)

    sz_bs = size(bs)
    #print(sz_bs)
    #print("\n")
    sz_bsd = size(bsd)
    #print(sz_bsd)
    #print("\n")


    #display("text/plain", bs)

    natoms = get_natoms(lmp)
    #@printf("%d atoms\n", natoms)
    return bs, natoms
end

function generate_bispectrum_matrix(path, rcut, twojmax, M, N1, N2)

    A = LMP(["-screen","none"]) do lmp
        A = Array{Float64}(undef, M, 2*ncoeff)
        for m in 1:M
            
            if (m%100==0)
                @printf "Config %d\n" m
            end
            data = joinpath(path, string(m), "DATA")
            bs, natoms = calc_bispectrum(lmp, data, rcut, twojmax)

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

function generate_target_array(path,M,N1,N2)

    y = Array{Float64}(undef, M, 1)
    #sz_y = size(y)
    #print(sz_y)
    #print("\n")

    for m in 1:M
        if (m%100==0)
            @printf "Config %d\n" m
        end
        open(joinpath(DATA, string(m), "PE")) do f
            s = split(readline(f))
            #print(s)
            r = parse(Float64, s[1])
            #print(r)
            #print("\n")
            #push!(y, r)
            y[m]=r
            #pe = parse(Float64, s)
            #print(pe)
        end
    end

    return y
end

print("Generating bispectrum matrix (A).\n")
A, ncoeff = generate_bispectrum_matrix(DATA, rcut, twojmax, M, N1, N2)
print("Generating target array (y).\n")
y = generate_target_array(DATA, M, N1, N2)
#sz_A = size(A)
#print(sz_A)
#print("\n")

# Solve for the coefficients.
print("Solving for coefficients (β).\n")
β = A \ y

# Write a LAMMPS snapcoeff file
open("fit.snapcoeff", "w") do io
    write(io, "#LAMMPS SNAP coeffs for hydrogen diffusing in palladium.\n") 
    write(io, "\n")
    println(io, ntypes, " ", ncoeff)
    println(io, "Pd ", r1, " ", w1)
    for b in 1:ncoeff
        println(io, β[b])
    end
    println(io, "H ", r2, " ", w2)
    for b in (ncoeff+1):(2*ncoeff)
        println(io, β[b])
    end
end
# Write a LAMMPS snapparam file
open("fit.snapparam", "w") do io
    write(io, "#LAMMPS SNAP coeffs for hydrogen diffusing in palladium.\n") 
    write(io, "\n")
    write(io, "#required\n")
    println(io, "rcutfac ", rcut)
    println(io, "twojmax ", twojmax)
    write(io, "\n")
    write(io, "#optional\n")
    println(io, "rfac0 ", 0.99363)
    println(io, "rmin0 ",  0)
    println(io, "bzeroflag ", 0)
    println(io, "quadraticflag ", 0)
end
