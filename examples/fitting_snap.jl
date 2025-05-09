# # Fitting SNAP using GaN data
#
# This example: 
#    - Uses the atomic positions in the folder `example_GaN_data`
#    - Generates surrogate DFT data based on the GaN model presented in `10.1088/1361-648x/ab6cbe`
#    - Uses `snap.jl` and 80% of the GaN data to create the matrix A. The matrix is generated only with the energy block.
#    - Uses 80% of the GaN data to create the vector b. The reference energy (``E_{\rm ref}``) is assumed to be zero.
#    - Uses backslash to fit the parameters ``\boldsymbol{\beta}``, thus, solves ``\mathbf{A \cdot \boldsymbol{\beta}=y}``
#    - Uses 20% of the GaN data and the bispectrum components provided by `snap.jl`and the fitted parameters ``\boldsymbol{\beta}`` to validate the fitting.
#    - The error computed during the validation is < ~5%.
# TODO:
#    - Check if the GaN model is correct.
#    - Calculate reference energy (``E_{\rm ref}``).
#    - Check ill-conditioned system.

#
# ## Install and import dependencies
#
# First let's make sure we have all required packages installed

# ```julia
# using Pkg
# pkg"add LAMMPS"
# ```
#
# We start off by importing the necessary packages

using MPI; MPI.Init()
using LAMMPS
using LinearAlgebra: norm, pinv
using Printf

# ##  Estimation of the SNAP coefficients
#
# The choice of the coefficients ``\boldsymbol{\beta}=(\beta_0^1, \tilde{\beta}^1, \dots, \beta_0^l, \tilde{\beta}^l)``
# is based on a system of linear equations which considers a large number of atomic
# configurations and ``l`` atom types. The matrix formulation for this system ``\mathbf{A \cdot \boldsymbol{\beta}=y}``
# is defined in the following equations (see 10.1016/j.jcp.2014.12.018):

# ```math
# \begin{equation*}
#    \mathbf{A}=
#    \begin{pmatrix}
#        \vdots &  &  & & & & & & \\
#        N_{s_1} & \sum_{i=1}^{N_{s_1}} B_1^i & \dots & \sum_{i=1}^{N_{s_1}} B_k^i  & \dots & N_{s_L} & \sum_{i=1}^{N_{s_L}} B_1^i & \dots & \sum_{i=1}^{N_{s_L}} B_k^i\\
#        \vdots &  &  & & & & & & \\
#        0 & -\sum_{i=1}^{N_{s_1}} \frac{\partial B_1^i}{\partial r_j^{\alpha}} & \dots & -\sum_{i=1}^{N_{s_1}} \frac{\partial B_k^i}{\partial r_j^{\alpha}} & \dots &  0 & -\sum_{i=1}^{N_{s_l}} \frac{\partial B_1^i}{\partial r_j^{\alpha}} & \dots & -\sum_{i=1}^{N_{s_l}} \frac{\partial B_k^i}{\partial r_j^{\alpha}} \\
#        \vdots &  &  & & & & & & \\
#        0 & - \sum_{j=1}^{N_{s_1}} r^j_{\alpha} \sum_{i=1}^{N_{s_1}} \frac{\partial B_1^i}{\partial r_j^{\beta}} & \dots & - \sum_{j=1}^{N_{s_1}} r^j_{\alpha} \sum_{i=1}^{N_{s_1}} \frac{\partial B_k^i}{\partial r_j^{\beta}} & \dots & 0 & - \sum_{j=1}^{N_{s_l}} r^j_{\alpha} \sum_{i=1}^{N_{s_l}} \frac{\partial B_1^i}{\partial r_j^{\beta}} & \dots & - \sum_{j=1}^{N_{s_l}} r^j_{\alpha} \sum_{i=1}^{N_{s_l}} \frac{\partial B_k^i}{\partial r_j^{\beta}}\\
#        \vdots &  &  & & & & & & \\
#    \end{pmatrix}
# \end{equation*}
# ```

# The indexes ``\alpha, \beta = 1,2,3`` depict the ``x``, ``y`` and ``z``
# spatial component, ``j`` is an individual atom, and ``s`` a particular configuration.
# All atoms in each configuration are considered. The number of atoms of type $m$ in the configuration
# ``s`` is ``N_{s_m}``.

# The RHS of the linear system is computed as:

# ```math
# \begin{equation*}
#  \mathbf{y}=  \begin{pmatrix}
#    \vdots \\
#   E^s_{\rm qm} -  E^s_{\rm ref} \\
#   \vdots \\\\
#   F^{s,j,\alpha}_{\rm qm} - F^{s,j,\alpha}_{\rm ref} \\
#   \vdots \\
#   W_{\rm qm}^{s,\alpha,\beta} - W_{\rm ref}^{s,\alpha,\beta} \\
#   \vdots \\        
#    \end{pmatrix}
# \end{equation*}
# ```


# # Calculate ``A`` using `snap.jl`

# This example only fits the energy, thus, the first block of the matrix A.

include(joinpath(dirname(pathof(LAMMPS)), "..", "examples", "snap.jl"))
A

# # Calculate ``y``

# A molecular mechanics model for the interaction of gallium and nitride is used
# to generate surrogate DFT data (see 10.1088/1361-648x/ab6cbe).
# The total potential energy is computed for each configuration

# ```math
# E = \sum_{i \lt j;  r_{i,j} \leq rcut } E_{GaN}(r_{i,j}) \  \text{ where } \\
#
# E_{GaN}(r_{i,j}) =
# \left\{
#    \begin{array}{ll}
#        E_{C}(r_{i,j}) + E_{LJ}(r_{i,j}, \epsilon_{Ga,Ga}, \sigma_{Ga,Ga})  & \mbox{if both atoms are Ga} \\
#        E_{C}(r_{i,j}) + E_{LJ}(r_{i,j}, \epsilon_{N,N}, \sigma_{N,N})  & \mbox{if both atoms are N} \\
#        E_{C}(r_{i,j}) + E_{BM}(r_{i,j}, A_{Ga,N}, \rho_{Ga,N})  & \mbox{if one atom is Ga and the other is N}\\
#    \end{array}
# \right.
# ```

# Note: this is a work in progress, this model should be checked.

const N = N1 + N2
const M1 = M
const M2 = 61

const ε_Ga_Ga = 0.643
const σ_Ga_Ga = 2.390
const ε_N_N = 1.474
const σ_N_N = 1.981
const A_Ga_N = 608.54
const ρ_Ga_N = 0.435
const q_Ga = 3.0
const q_N = -3.0
const ε0 = 55.26349406 # e2⋅GeV−1⋅fm−1 ToDo: check this
const E_ref = zeros(M1)
E_LJ(r, ε = 1.0, σ = 1.0) = 4.0 * ε * ((σ / norm(r))^12 - (σ / norm(r))^6)
E_BM(r, A = 1.0, ρ = 1.0) = A * exp(-norm(r) / ρ)
E_C(r) = q_Ga * q_N / (4.0 * π * ε0 * norm(r))

function read_atomic_conf(m, N)
    rs = []
    open(joinpath(DATA, string(m), "DATA")) do f
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

# Vector ``y`` is calculated using the DFT data and ``E_{\rm ref}``

function calc_y(rcut, M1, N, ε_Ga_Ga, σ_Ga_Ga, ε_N_N, σ_N_N, A_Ga_N, ρ_Ga_N, E_ref)
    y = zeros(M1)
    for m = 1:M1
        rs = read_atomic_conf(m, N)
        y[m] = calc_tot_energy(rcut, rs, N, ε_Ga_Ga, σ_Ga_Ga,
                               ε_N_N, σ_N_N, A_Ga_N, ρ_Ga_N) - E_ref[m]
    end
    return y
end

y = calc_y(rcut, M1, N, ε_Ga_Ga, σ_Ga_Ga, ε_N_N, σ_N_N, A_Ga_N, ρ_Ga_N, E_ref)

# ## Calculate the optimal solution

# The optimal solution $\widehat{\mathbf{\beta}}$ is, thus:
# ```math
# \begin{equation*}
#     \widehat{\mathbf{\beta}} = \mathrm{argmin}_{\mathbf{\beta}} ||\mathbf{A \cdot \boldsymbol{\beta} - y}||^2 = \mathbf{A^{-1} \cdot y}
# \end{equation*}
# ```

β = A \ y

# ## Check results

# The local energy can be decomposed into separate contributions for each atom.
# SNAP energy can be written in terms of the bispectrum components of the atoms
# and a set of coefficients. ``K`` components of the bispectrum are considered
# so that ``\mathbf{B}^{i}=\{ B^i_1, \dots, B_K^i\}`` for each atom ``i``, whose
# SNAP energy is computed as follows:
# ```math
#    E^i_{\rm SNAP}(\mathbf{B}^i) = \beta_0^{\alpha_i} + \sum_{k=1}^K \beta_k^{\alpha_i} B_k^i =  \beta_0^{\alpha_i} + \boldsymbol{\tilde{\beta}}^{\alpha_i} \cdot \mathbf{B}^i
# ```
# where $\alpha_i$ depends on the atom type.

function calc_fitted_tot_energy(path, β, ncoeff, N1, N)
    ## Calculate y
    lmp = LMP(["-screen","none"])
    bs = run_snap(lmp, path, rcut, twojmax)

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


# Comparison between the potential energy calculated based on the DFT data, and
# the SNAP potential energy calculated based on the bispectrum components
# provided by ``snap.jl`` and the fitted coefficients ``\mathbf{\beta}``.


@printf("Potential Energy, Fitted Potential Energy, Error (%%)\n")
for m = M1+1:M2
    path = joinpath(DATA, string(m), "DATA")
    rs = read_atomic_conf(m, N)
    E_tot = calc_tot_energy(rcut, rs, N, ε_Ga_Ga, σ_Ga_Ga, ε_N_N, σ_N_N, A_Ga_N, ρ_Ga_N)
    E_tot_fit = calc_fitted_tot_energy(path, β, ncoeff, N1, N)
    @printf("%0.2f, %0.2f, %0.2f\n", E_tot, E_tot_fit,
            abs(E_tot - E_tot_fit) / E_tot * 100.)
end


