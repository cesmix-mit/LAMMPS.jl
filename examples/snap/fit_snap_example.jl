using LinearAlgebra: norm, pinv
using GalacticOptim, Optim
using Printf

################################################################################
# SNAP fitting example using `snap.jl`
# Solve A β = b
# See: https://www.sciencedirect.com/science/article/pii/S0021999114008353
################################################################################

# Calculate A ##################################################################
include("snap.jl")
const N1 = 96
const N2 = 96
const N = N1 + N2
const M1 = 48
const M2 = 61

# Calculate b ##################################################################

# GaN model, used as a surrogate for DFT data. ToDo: check model
# Reference: https://iopscience.iop.org/article/10.1088/1361-648X/ab6cbe
const ε_Ga_Ga = 0.643
const σ_Ga_Ga = 2.390
const ε_N_N = 1.474
const σ_N_N = 1.981
const A_Ga_N = 608.54
const ρ_Ga_N = 0.435
const q_Ga = 3.0
const q_N = -3.0
const ε0 = 55.26349406 # e2⋅GeV−1⋅fm−1 ToDo: check this
const E_ref = 0.
E_LJ(r, ε = 1.0, σ = 1.0) = 4.0 * ε * ((σ / norm(r))^12 - (σ / norm(r))^6)
E_BM(r, A = 1.0, ρ = 1.0) = A * exp(-norm(r) / ρ)
E_C(r) = q_Ga * q_N / (4.0 * π * ε0 * norm(r))


# Calc. total potential energy for each configuration

function read_atomic_conf(m, N)
    rs = []
    open(string("data/", string(m), "/DATA")) do f
        for i = 1:23
            readline(f)
        end
        for i = 1:N
            s = split(readline(f))
            r = [parse(Float64, s[3]),
                 parse(Float64, s[4]),
                 parse(Float64, s[5])]
            push!(rs, r)
        end
    end
    return rs
end

function calc_tot_energy(rcut, rs, N, ε_Ga_Ga, σ_Ga_Ga, ε_N_N, σ_N_N, A_Ga_N, ρ_Ga_N)
    E_tot_acc = 0.0
    for i = 1:N
        for j = i:N
            r_diff = rs[i] - rs[j]
            if norm(r_diff) <= rcut && norm(r_diff) > 0.0
                if i <= N1 && j <= N1 
                    E_tot_acc += E_C(r_diff) + E_LJ(r_diff, ε_Ga_Ga, σ_Ga_Ga)
                elseif i > N1 && j > N1 
                    E_tot_acc += E_C(r_diff) + E_LJ(r_diff, ε_N_N, σ_N_N)
                else
                    E_tot_acc += E_C(r_diff) + E_BM(r_diff, A_Ga_N, ρ_Ga_N)
                end
            end  
        end
    end
    return E_tot_acc
end

function calc_b(rcut, M1, N, ε_Ga_Ga, σ_Ga_Ga, ε_N_N, σ_N_N, A_Ga_N, ρ_Ga_N)
    b = zeros(M1)
    for m = 1:M1
        rs = read_atomic_conf(m, N)
        b[m] = calc_tot_energy(rcut, rs, N, ε_Ga_Ga, σ_Ga_Ga,
                               ε_N_N, σ_N_N, A_Ga_N, ρ_Ga_N)
    end
    return b
end

b = calc_b(rcut, M1, N, ε_Ga_Ga, σ_Ga_Ga, ε_N_N, σ_N_N, A_Ga_N, ρ_Ga_N)


# Calculate β ##################################################################

#β = A \ b

cost_function(β, p) = norm(A * β - b)
β0 = zeros(2 * ncoeff)
prob = OptimizationProblem(cost_function, β0)
β = solve(prob, NelderMead(), maxiters=2000)


# Check results ################################################################

function calc_fitted_tot_energy(path, β, ncoeff, N1, N)
    ## Calculate b
    lmp = LMP(["-screen","none"]) 
    read_data_str = string("read_data ", path)

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
    command(lmp, "compute SNA all sna/atom $rcut 0.99363 $twojmax 0.5 0.5 1.0 0.5 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1")
    command(lmp, "compute SNAD all snad/atom $rcut 0.99363 $twojmax 0.5 0.5 1.0 0.5 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1")
    command(lmp, "compute SNAV all snav/atom $rcut 0.99363 $twojmax 0.5 0.5 1.0 0.5 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1")
    command(lmp, "thermo_style custom pe")
    #command(lmp, "dump 2 all custom 100 dump.forces fx fy fz")
    command(lmp, "run 0")
    nlocal = extract_global(lmp, "nlocal")
    types = extract_atom(lmp, "type", LAMMPS.API.LAMMPS_INT)
    ids = extract_atom(lmp, "id", LAMMPS.API.LAMMPS_INT)
    bs = extract_compute(lmp, "SNA", LAMMPS.API.LMP_STYLE_ATOM,
                                     LAMMPS.API.LMP_TYPE_ARRAY)
    

    E_tot_acc = 0.0
    for n in 1:N1
        E_atom_acc = β[1]
        for k in 2:ncoeff
            k2 = k - 1
            E_atom_acc += β[k] * bs[k2, n]
        end
        E_tot_acc += E_atom_acc
    end
    for n in N1+1:N
        E_atom_acc = β[ncoeff+1]
        for k in ncoeff+2:2*ncoeff
            k2 = k - ncoeff - 1
            E_atom_acc += β[k] * bs[k2, n]
        end
        E_tot_acc += E_atom_acc
    end
    command(lmp, "clear")
    return E_tot_acc
end

@printf("Potential Energy, Fitted Potential Energy, Error (%%)\n")
for m = M1+1:M2
    path = string("data/", string(m), "/DATA")
    rs = read_atomic_conf(m, N)
    E_tot = calc_tot_energy(rcut, rs, N, ε_Ga_Ga, σ_Ga_Ga, ε_N_N, σ_N_N, A_Ga_N, ρ_Ga_N)
    E_tot_fit = calc_fitted_tot_energy(path, β, ncoeff, N1, N)
    @printf("%0.2f, %0.2f, %0.2f\n", E_tot, E_tot_fit,
            abs(E_tot - E_tot_fit) / E_tot * 100.)
end

