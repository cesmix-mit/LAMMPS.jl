var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Modules = [LAMMPS]","category":"page"},{"location":"api/#LAMMPS.LMP","page":"API","title":"LAMMPS.LMP","text":"LMP(f::Function, args=String[], comm=nothing)\n\nCreate a new LAMMPS instance and call f on that instance while returning the result from f.\n\n\n\n\n\n","category":"type"},{"location":"api/#LAMMPS.LMP-2","page":"API","title":"LAMMPS.LMP","text":"LMP(args::Vector{String}=String[], comm::Union{Nothing, MPI.Comm}=nothing)\n\nCreate a new LAMMPS instance while passing in a list of strings as if they were command-line arguments for the LAMMPS executable.\n\nFor a full ist of Command-line options see: https://docs.lammps.org/Run_options.html\n\n\n\n\n\n","category":"type"},{"location":"api/#LAMMPS.close!-Tuple{LMP}","page":"API","title":"LAMMPS.close!","text":"close!(lmp::LMP)\n\nShutdown a LAMMPS instance.\n\n\n\n\n\n","category":"method"},{"location":"api/#LAMMPS.command-Tuple{LMP, Union{String, Array{String}}}","page":"API","title":"LAMMPS.command","text":"command(lmp::LMP, cmd::Union{String, Array{String}})\n\nProcess LAMMPS input commands from a String or from an Array of Strings.\n\nFor a full list of commands see: https://docs.lammps.org/commands_list.html\n\nThis function processes a multi-line string similar to a block of commands from a file. The string may have multiple lines (separated by newline characters) and also single commands may be distributed over multiple lines with continuation characters (’&’). Those lines are combined by removing the ‘&’ and the following newline character. After this processing the string is handed to LAMMPS for parsing and executing.\n\nArrays of Strings get concatenated into a single String inserting newline characters as needed.\n\ncompat: LAMMPS.jl 0.4.1\nMultiline string support \"\"\" and support for array of strings was added. Prior versions of LAMMPS.jl ignore newline characters.\n\nExamples\n\nLMP([\"-screen\", \"none\"]) do lmp\n    command(lmp, \"\"\"\n        atom_modify map yes\n        region cell block 0 2 0 2 0 2\n        create_box 1 cell\n        lattice sc 1\n        create_atoms 1 region cell\n        mass 1 1\n\n        group a id 1 2 3 5 8\n        group even id 2 4 6 8\n        group odd id 1 3 5 7\n    \"\"\")\nend\n\n\n\n\n\n","category":"method"},{"location":"api/#LAMMPS.extract_atom-Tuple{LMP, String, LAMMPS._LMP_DATATYPE}","page":"API","title":"LAMMPS.extract_atom","text":"extract_atom(lmp::LMP, name::String, lmp_type::_LMP_DATATYPE; copy=false)\n\nExtract per-atom data from the lammps instance.\n\nvalid values for lmp_type: resulting return type:\nLAMMPS_INT Vector{Int32}\nLAMMPS_INT_2D Matrix{Int32}\nLAMMPS_DOUBLE Vector{Float64}\nLAMMPS_DOUBLE_2D Matrix{Float64}\nLAMMPS_INT64 Vector{Int64}\nLAMMPS_INT64_2D Matrix{Int64}\n\nthe kwarg copy, which defaults to true, determies wheter a copy of the underlying data is made. As the pointer to the underlying data is not persistent, it's highly recommended to only disable this, if you wish to modify the internal state of the LAMMPS instance.\n\ninfo: Info\nThe returned data may become invalid if a re-neighboring operation is triggered at any point after calling this method. If this has happened, trying to read from this data will likely cause julia to crash. To prevent this, set copy=true.\n\nA table with suported name keywords can be found here: https://docs.lammps.org/Classes_atom.html#_CPPv4N9LAMMPS_NS4Atom7extractEPKc\n\n\n\n\n\n","category":"method"},{"location":"api/#LAMMPS.extract_compute-Tuple{LMP, String, LAMMPS.API._LMP_STYLE_CONST, LAMMPS._LMP_TYPE}","page":"API","title":"LAMMPS.extract_compute","text":"extract_compute(lmp::LMP, name::String, style::_LMP_STYLE_CONST, lmp_type::_LMP_TYPE; copy::Bool=true)\n\nExtract data provided by a compute command identified by the compute-ID. Computes may provide global, per-atom, or local data, and those may be a scalar, a vector or an array. Since computes may provide multiple kinds of data, it is required to set style and type flags representing what specific data is desired.\n\nvalid values for style:\nSTYLE_GLOBAL\nSTYLE_ATOM\nSTYLE_LOCAL\n\nvalid values for lmp_type: resulting return type:\nTYPE_SCALAR Vector{Float64}\nTYPE_VECTOR Vector{Float64}\nTYPE_ARRAY Matrix{Float64}\nSIZE_VECTOR Vector{Int32}\nSIZE_COLS Vector{Int32}\nSIZE_ROWS Vector{Int32}\n\nScalar values get returned as a vector with a single element. This way it's possible to modify the internal state of the LAMMPS instance even if the data is scalar.\n\ninfo: Info\nThe returned data may become invalid as soon as another LAMMPS command has been issued at any point after calling this method. If this has happened, trying to read from this data will likely cause julia to crash. To prevent this, set copy=true.\n\nExamples\n\n    LMP([\"-screen\", \"none\"]) do lmp\n        extract_compute(lmp, \"thermo_temp\", LMP_STYLE_GLOBAL, TYPE_VECTOR, copy=true)[2] = 2\n        extract_compute(lmp, \"thermo_temp\", LMP_STYLE_GLOBAL, TYPE_VECTOR, copy=false)[3] = 3\n\n        extract_compute(lmp, \"thermo_temp\", LMP_STYLE_GLOBAL, TYPE_SCALAR) |> println # [0.0]\n        extract_compute(lmp, \"thermo_temp\", LMP_STYLE_GLOBAL, TYPE_VECTOR) |> println # [0.0, 0.0, 3.0, 0.0, 0.0, 0.0]\n    end\n\n\n\n\n\n","category":"method"},{"location":"api/#LAMMPS.extract_global-Tuple{LMP, String, LAMMPS._LMP_DATATYPE}","page":"API","title":"LAMMPS.extract_global","text":"extract_global(lmp::LMP, name::String, lmp_type::_LMP_DATATYPE; copy::Bool=false)\n\nExtract a global property from a LAMMPS instance.\n\nvalid values for lmp_type: resulting return type:\nLAMMPS_INT Vector{Int32}\nLAMMPS_INT_2D Matrix{Int32}\nLAMMPS_DOUBLE Vector{Float64}\nLAMMPS_DOUBLE_2D Matrix{Float64}\nLAMMPS_INT64 Vector{Int64}\nLAMMPS_INT64_2D Matrix{Int64}\nLAMMPS_STRING String (allways a copy)\n\nScalar values get returned as a vector with a single element. This way it's possible to modify the internal state of the LAMMPS instance even if the data is scalar.\n\ninfo: Info\nClosing the LAMMPS instance or issuing a clear command after calling this method will result in the returned data becoming invalid. To prevent this, set copy=true.\n\nwarning: Warning\nModifying the data through extract_global may lead to inconsistent internal data and thus may cause failures or crashes or bogus simulations. In general it is thus usually better to use a LAMMPS input command that sets or changes these parameters. Those will take care of all side effects and necessary updates of settings derived from such settings.\n\nA full list of global variables can be found here: https://docs.lammps.org/Library_properties.html\n\n\n\n\n\n","category":"method"},{"location":"api/#LAMMPS.extract_setting-Tuple{LMP, String}","page":"API","title":"LAMMPS.extract_setting","text":"extract_setting(lmp::LMP, name::String)::Int32\n\nQuery LAMMPS about global settings.\n\nA full list of settings can be found here: https://docs.lammps.org/Library_properties.html\n\nExamples\n\n    LMP([\"-screen\", \"none\"]) do lmp\n        command(lmp, \"\"\"\n            region cell block 0 3 0 3 0 3\n            create_box 1 cell\n            lattice sc 1\n            create_atoms 1 region cell\n        \"\"\")\n\n        extract_setting(lmp, \"dimension\") |> println # 3\n        extract_setting(lmp, \"nlocal\") |> println # 27\n    end\n\n\n\n\n\n","category":"method"},{"location":"api/#LAMMPS.extract_variable","page":"API","title":"LAMMPS.extract_variable","text":"extract_variable(lmp::LMP, name::String, lmp_variable::LMP_VARIABLE, group::Union{String, Nothing}=nothing; copy::Bool=false)\n\nExtracts the data from a LAMMPS variable. When the variable is either an equal-style compatible variable, a vector-style variable, or an atom-style variable, the variable is evaluated and the corresponding value(s) returned. Variables of style internal are compatible with equal-style variables, if they return a numeric value. For other variable styles, their string value is returned.\n\nvalid values for lmp_variable: return type\nVAR_ATOM Vector{Float64}\nVAR_EQUAL Float64\nVAR_STRING String\nVAR_VECTOR Vector{Float64}\n\nthe kwarg copy determies wheter a copy of the underlying data is made. copy is only aplicable for VAR_VECTOR and VAR_ATOM. For all other variable types, a copy will be made regardless. The underlying LAMMPS API call for VAR_ATOM internally allways creates a copy of the data. As the memory for this gets allocated by LAMMPS instead of julia, it needs to be dereferenced using LAMMPS.API.lammps_free instead of through the garbage collector.  If copy=false this gets acieved by registering LAMMPS.API.lammps_free as a finalizer for the returned data. Alternatively, setting copy=true will instead create a new copy of the data. The lammps allocated block of memory will then be freed immediately.\n\nthe kwarg group determines for which atoms the variable will be extracted. It's only aplicable for VAR_ATOM and will cause an error if used for other variable types. The entires for all atoms not in the group will be zeroed out. By default, all atoms will be extracted.\n\n\n\n\n\n","category":"function"},{"location":"api/#LAMMPS.gather","page":"API","title":"LAMMPS.gather","text":"gather(lmp::LMP, name::String, T::Union{Type{Int32}, Type{Float64}}, ids::Union{Nothing, Array{Int32}}=nothing)\n\nGather the named per-atom, per-atom fix, per-atom compute, or fix property/atom-based entities from all processes. By default (when ids=nothing), this method collects data from all atoms in consecutive order according to their IDs. The optional parameter ids determines for which subset of atoms the requested data will be gathered. The returned data will then be ordered according to ids\n\nCompute entities have the prefix c_, fix entities use the prefix f_, and per-atom entites have no prefix.\n\nThe returned Array is decoupled from the internal state of the LAMMPS instance.\n\nwarning: Type Verification\nDue to how the underlying C-API works, it's not possible to verify the element data-type of fix or compute style data. Supplying the wrong data-type will not throw an error but will result in nonsensical output\n\nwarning: ids\nThe optional parameter ids only works, if there is a map defined. For example by doing: command(lmp, \"atom_modify map yes\") However, LAMMPS only issues a warning if that's the case, which unfortuately cannot be detected through the underlying API. Starting form LAMMPS version 17 Apr 2024 this should no longer be an issue, as LAMMPS then throws an error instead of a warning.\n\n\n\n\n\n","category":"function"},{"location":"api/#LAMMPS.get_category_ids","page":"API","title":"LAMMPS.get_category_ids","text":"get_category_ids(lmp::LMP, category::String, buffer_size::Integer=50)\n\nLook up the names of entities within a certain category.\n\nValid categories are: compute, dump, fix, group, molecule, region, and variable. names longer than buffer_size will be truncated to fit inside the buffer.\n\n\n\n\n\n","category":"function"},{"location":"api/#LAMMPS.get_natoms-Tuple{LMP}","page":"API","title":"LAMMPS.get_natoms","text":"get_natoms(lmp::LMP)::Int64\n\nGet the total number of atoms in the LAMMPS instance.\n\nWill be precise up to 53-bit signed integer due to the underlying lammps_get_natoms returning a Float64.\n\n\n\n\n\n","category":"method"},{"location":"api/#LAMMPS.group_to_atom_ids-Tuple{LMP, String}","page":"API","title":"LAMMPS.group_to_atom_ids","text":"group_to_atom_ids(lmp::LMP, group::String)\n\nFind the IDs of the Atoms in the group.\n\n\n\n\n\n","category":"method"},{"location":"api/#LAMMPS.locate-Tuple{}","page":"API","title":"LAMMPS.locate","text":"locate()\n\nLocate the LAMMPS library currently being used, by LAMMPS.jl\n\n\n\n\n\n","category":"method"},{"location":"api/#LAMMPS.scatter!-Union{Tuple{T}, Tuple{LMP, String, VecOrMat{T}}, Tuple{LMP, String, VecOrMat{T}, Union{Nothing, Array{Int32}}}} where T<:Union{Float64, Int32}","page":"API","title":"LAMMPS.scatter!","text":"scatter!(lmp::LMP, name::String, data::VecOrMat{T}, ids::Union{Nothing, Array{Int32}}=nothing) where T<:Union{Int32, Float64}\n\nScatter the named per-atom, per-atom fix, per-atom compute, or fix property/atom-based entity in data to all processes. By default (when ids=nothing), this method scatters data to all atoms in consecutive order according to their IDs. The optional parameter ids determines to which subset of atoms the data will be scattered.\n\nCompute entities have the prefix c_, fix entities use the prefix f_, and per-atom entites have no prefix.\n\nwarning: Type Verification\nDue to how the underlying C-API works, it's not possible to verify the element data-type of fix or compute style data. Supplying the wrong data-type will not throw an error but will result in nonsensical date being supplied to the LAMMPS instance.\n\nwarning: ids\nThe optional parameter ids only works, if there is a map defined. For example by doing: command(lmp, \"atom_modify map yes\") However, LAMMPS only issues a warning if that's the case, which unfortuately cannot be detected through the underlying API. Starting form LAMMPS version 17 Apr 2024 this should no longer be an issue, as LAMMPS then throws an error instead of a warning.\n\n\n\n\n\n","category":"method"},{"location":"api/#LAMMPS.set_library!-Tuple{Any}","page":"API","title":"LAMMPS.set_library!","text":"set_library!(path)\n\nChange the library path used by LAMMPS.jl for liblammps.so to path.\n\nnote: Note\nYou will need to restart Julia to use the new library.\n\nwarning: Warning\nDue to a bug in Julia (until 1.6.5 and 1.7.1), setting preferences in transitive dependencies is broken (https://github.com/JuliaPackaging/Preferences.jl/issues/24). To fix this either update your version of Julia, or add LAMMPS_jll as a direct dependency to your project.\n\n\n\n\n\n","category":"method"},{"location":"generated/snap/","page":"Basic SNAP","title":"Basic SNAP","text":"EditURL = \"../../../examples/snap.jl\"","category":"page"},{"location":"generated/snap/#SNAP-example-using-GaN","page":"Basic SNAP","title":"SNAP example using GaN","text":"","category":"section"},{"location":"generated/snap/","page":"Basic SNAP","title":"Basic SNAP","text":"In this example we extract the bispectrum from SNAP. This example demonstrates how to install and use LAMMPS.jl","category":"page"},{"location":"generated/snap/#Install-dependencies","page":"Basic SNAP","title":"Install dependencies","text":"","category":"section"},{"location":"generated/snap/","page":"Basic SNAP","title":"Basic SNAP","text":"First let's make sure we have all required packages installed","category":"page"},{"location":"generated/snap/","page":"Basic SNAP","title":"Basic SNAP","text":"using Pkg\npkg\"add LAMMPS\"","category":"page"},{"location":"generated/snap/","page":"Basic SNAP","title":"Basic SNAP","text":"LAMMPS.jl will automatically download and install a pre-built version of LAMMPS.","category":"page"},{"location":"generated/snap/","page":"Basic SNAP","title":"Basic SNAP","text":"We start off by importing LAMMPS.","category":"page"},{"location":"generated/snap/","page":"Basic SNAP","title":"Basic SNAP","text":"using LAMMPS\n\nconst DATA = joinpath(dirname(pathof(LAMMPS)), \"..\", \"examples\", \"example_GaN_data\")\n\nfunction run_snap(lmp, path, rcut, twojmax)\n    command(lmp, \"\"\"\n        log none\n        units metal\n        boundary p p p\n        atom_style atomic\n        atom_modify map array\n        read_data $path\n        pair_style zero $rcut\n        pair_coeff * *\n        compute PE all pe\n        compute S all pressure thermo_temp\n        compute SNA all sna/atom $rcut 0.99363 $twojmax 0.5 0.5 1.0 0.5 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1\n        compute SNAD all snad/atom $rcut 0.99363 $twojmax 0.5 0.5 1.0 0.5 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1\n        compute SNAV all snav/atom $rcut 0.99363 $twojmax 0.5 0.5 1.0 0.5 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1\n        thermo_style custom pe\n        run 0\n    \"\"\")\n\n    # Extract bispectrum\n    bs = gather(lmp, \"c_SNA\", Float64)\n    return bs\nend\n\nfunction calculate_snap_bispectrum(path, rcut, twojmax, M, N1, N2)\n    J = twojmax / 2\n    ncoeff = round(Int, (J+1)*(J+2)*((J+(1.5))/3) + 1)\n\n    A = LMP([\"-screen\",\"none\"]) do lmp\n        A = Array{Float64}(undef, M, 2*ncoeff) # bispectrum is 2*(ncoeff - 1) + 2\n        for m in 1:M\n            data = joinpath(path, string(m), \"DATA\")\n            bs = run_snap(lmp, data, rcut, twojmax)\n\n            # Make bispectrum sum vector\n            row = Float64[]\n\n            push!(row, N1)\n\n            for k in 1:(ncoeff-1)\n                acc = 0.0\n                for n in 1:N1\n                    acc += bs[k, n]\n                end\n                push!(row, acc)\n            end\n\n            push!(row, N2)\n            for k in 1:(ncoeff-1)\n                acc = 0.0\n                for n in N1 .+ (1:N2)\n                    acc += bs[k, n]\n                end\n                push!(row, acc)\n            end\n\n            A[m, :] = row\n\n            command(lmp, \"clear\")\n        end\n        return A\n    end\n    return A, ncoeff\nend\n\nconst rcut = 3.5\nconst twojmax = 6\nconst M  = 48 # number of input files\nconst N1 = 96 # number of atoms of the first type\nconst N2 = 96 # number of atoms of the second type\nA, ncoeff = calculate_snap_bispectrum(DATA, rcut, twojmax, M, N1, N2)","category":"page"},{"location":"generated/snap/","page":"Basic SNAP","title":"Basic SNAP","text":"","category":"page"},{"location":"generated/snap/","page":"Basic SNAP","title":"Basic SNAP","text":"This page was generated using Literate.jl.","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"EditURL = \"../../../examples/fitting_snap.jl\"","category":"page"},{"location":"generated/fitting_snap/#Fitting-SNAP-using-GaN-data","page":"Fitting SNAP","title":"Fitting SNAP using GaN data","text":"","category":"section"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"This example:","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"Uses the atomic positions in the folder example_GaN_data\nGenerates surrogate DFT data based on the GaN model presented in 10.1088/1361-648x/ab6cbe\nUses snap.jl and 80% of the GaN data to create the matrix A. The matrix is generated only with the energy block.\nUses 80% of the GaN data to create the vector b. The reference energy (E_rm ref) is assumed to be zero.\nUses backslash to fit the parameters boldsymbolbeta, thus, solves mathbfA cdot boldsymbolbeta=y\nUses 20% of the GaN data and the bispectrum components provided by snap.jland the fitted parameters boldsymbolbeta to validate the fitting.\nThe error computed during the validation is < ~5%.","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"TODO:","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"Check if the GaN model is correct.\nCalculate reference energy (E_rm ref).\nCheck ill-conditioned system.","category":"page"},{"location":"generated/fitting_snap/#Install-and-import-dependencies","page":"Fitting SNAP","title":"Install and import dependencies","text":"","category":"section"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"First let's make sure we have all required packages installed","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"using Pkg\npkg\"add LAMMPS\"","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"We start off by importing the necessary packages","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"using LAMMPS\nusing LinearAlgebra: norm, pinv\nusing Printf","category":"page"},{"location":"generated/fitting_snap/#Estimation-of-the-SNAP-coefficients","page":"Fitting SNAP","title":"Estimation of the SNAP coefficients","text":"","category":"section"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"The choice of the coefficients boldsymbolbeta=(beta_0^1 tildebeta^1 dots beta_0^l tildebeta^l) is based on a system of linear equations which considers a large number of atomic configurations and l atom types. The matrix formulation for this system mathbfA cdot boldsymbolbeta=y is defined in the following equations (see 10.1016/j.jcp.2014.12.018):","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"beginequation*\n   mathbfA=\n   beginpmatrix\n       vdots           \n       N_s_1  sum_i=1^N_s_1 B_1^i  dots  sum_i=1^N_s_1 B_k^i   dots  N_s_L  sum_i=1^N_s_L B_1^i  dots  sum_i=1^N_s_L B_k^i\n       vdots           \n       0  -sum_i=1^N_s_1 fracpartial B_1^ipartial r_j^alpha  dots  -sum_i=1^N_s_1 fracpartial B_k^ipartial r_j^alpha  dots   0  -sum_i=1^N_s_l fracpartial B_1^ipartial r_j^alpha  dots  -sum_i=1^N_s_l fracpartial B_k^ipartial r_j^alpha \n       vdots           \n       0  - sum_j=1^N_s_1 r^j_alpha sum_i=1^N_s_1 fracpartial B_1^ipartial r_j^beta  dots  - sum_j=1^N_s_1 r^j_alpha sum_i=1^N_s_1 fracpartial B_k^ipartial r_j^beta  dots  0  - sum_j=1^N_s_l r^j_alpha sum_i=1^N_s_l fracpartial B_1^ipartial r_j^beta  dots  - sum_j=1^N_s_l r^j_alpha sum_i=1^N_s_l fracpartial B_k^ipartial r_j^beta\n       vdots           \n   endpmatrix\nendequation*","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"The indexes alpha beta = 123 depict the x, y and z spatial component, j is an individual atom, and s a particular configuration. All atoms in each configuration are considered. The number of atoms of type m in the configuration s is N_s_m.","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"The RHS of the linear system is computed as:","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"beginequation*\n mathbfy=  beginpmatrix\n   vdots \n  E^s_rm qm -  E^s_rm ref \n  vdots \n  F^sjalpha_rm qm - F^sjalpha_rm ref \n  vdots \n  W_rm qm^salphabeta - W_rm ref^salphabeta \n  vdots \n   endpmatrix\nendequation*","category":"page"},{"location":"generated/fitting_snap/#Calculate-A-using-snap.jl","page":"Fitting SNAP","title":"Calculate A using snap.jl","text":"","category":"section"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"This example only fits the energy, thus, the first block of the matrix A.","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"include(joinpath(dirname(pathof(LAMMPS)), \"..\", \"examples\", \"snap.jl\"))\nA","category":"page"},{"location":"generated/fitting_snap/#Calculate-y","page":"Fitting SNAP","title":"Calculate y","text":"","category":"section"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"A molecular mechanics model for the interaction of gallium and nitride is used to generate surrogate DFT data (see 10.1088/1361-648x/ab6cbe). The total potential energy is computed for each configuration","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"E = sum_i lt j  r_ij leq rcut  E_GaN(r_ij)   text where  \n\nE_GaN(r_ij) =\nleft\n   beginarrayll\n       E_C(r_ij) + E_LJ(r_ij epsilon_GaGa sigma_GaGa)   mboxif both atoms are Ga \n       E_C(r_ij) + E_LJ(r_ij epsilon_NN sigma_NN)   mboxif both atoms are N \n       E_C(r_ij) + E_BM(r_ij A_GaN rho_GaN)   mboxif one atom is Ga and the other is N\n   endarray\nright","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"Note: this is a work in progress, this model should be checked.","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"const N = N1 + N2\nconst M1 = M\nconst M2 = 61\n\nconst ε_Ga_Ga = 0.643\nconst σ_Ga_Ga = 2.390\nconst ε_N_N = 1.474\nconst σ_N_N = 1.981\nconst A_Ga_N = 608.54\nconst ρ_Ga_N = 0.435\nconst q_Ga = 3.0\nconst q_N = -3.0\nconst ε0 = 55.26349406 # e2⋅GeV−1⋅fm−1 ToDo: check this\nconst E_ref = zeros(M1)\nE_LJ(r, ε = 1.0, σ = 1.0) = 4.0 * ε * ((σ / norm(r))^12 - (σ / norm(r))^6)\nE_BM(r, A = 1.0, ρ = 1.0) = A * exp(-norm(r) / ρ)\nE_C(r) = q_Ga * q_N / (4.0 * π * ε0 * norm(r))\n\nfunction read_atomic_conf(m, N)\n    rs = []\n    open(joinpath(DATA, string(m), \"DATA\")) do f\n        for i = 1:23\n            readline(f)\n        end\n        for i = 1:N\n            s = split(readline(f))\n            r = [parse(Float64, s[3]),\n                 parse(Float64, s[4]),\n                 parse(Float64, s[5])]\n            push!(rs, r)\n        end\n    end\n    return rs\nend\n\nfunction calc_tot_energy(rcut, rs, N, ε_Ga_Ga, σ_Ga_Ga, ε_N_N, σ_N_N, A_Ga_N, ρ_Ga_N)\n    E_tot_acc = 0.0\n    for i = 1:N\n        for j = i:N\n            r_diff = rs[i] - rs[j]\n            if norm(r_diff) <= rcut && norm(r_diff) > 0.0\n                if i <= N1 && j <= N1\n                    E_tot_acc += E_C(r_diff) + E_LJ(r_diff, ε_Ga_Ga, σ_Ga_Ga)\n                elseif i > N1 && j > N1\n                    E_tot_acc += E_C(r_diff) + E_LJ(r_diff, ε_N_N, σ_N_N)\n                else\n                    E_tot_acc += E_C(r_diff) + E_BM(r_diff, A_Ga_N, ρ_Ga_N)\n                end\n            end\n        end\n    end\n    return E_tot_acc\nend","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"Vector y is calculated using the DFT data and E_rm ref","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"function calc_y(rcut, M1, N, ε_Ga_Ga, σ_Ga_Ga, ε_N_N, σ_N_N, A_Ga_N, ρ_Ga_N, E_ref)\n    y = zeros(M1)\n    for m = 1:M1\n        rs = read_atomic_conf(m, N)\n        y[m] = calc_tot_energy(rcut, rs, N, ε_Ga_Ga, σ_Ga_Ga,\n                               ε_N_N, σ_N_N, A_Ga_N, ρ_Ga_N) - E_ref[m]\n    end\n    return y\nend\n\ny = calc_y(rcut, M1, N, ε_Ga_Ga, σ_Ga_Ga, ε_N_N, σ_N_N, A_Ga_N, ρ_Ga_N, E_ref)","category":"page"},{"location":"generated/fitting_snap/#Calculate-the-optimal-solution","page":"Fitting SNAP","title":"Calculate the optimal solution","text":"","category":"section"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"The optimal solution widehatmathbfbeta is, thus:","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"beginequation*\n    widehatmathbfbeta = mathrmargmin_mathbfbeta mathbfA cdot boldsymbolbeta - y^2 = mathbfA^-1 cdot y\nendequation*","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"β = A \\ y","category":"page"},{"location":"generated/fitting_snap/#Check-results","page":"Fitting SNAP","title":"Check results","text":"","category":"section"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"The local energy can be decomposed into separate contributions for each atom. SNAP energy can be written in terms of the bispectrum components of the atoms and a set of coefficients. K components of the bispectrum are considered so that mathbfB^i= B^i_1 dots B_K^i for each atom i, whose SNAP energy is computed as follows:","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"   E^i_rm SNAP(mathbfB^i) = beta_0^alpha_i + sum_k=1^K beta_k^alpha_i B_k^i =  beta_0^alpha_i + boldsymboltildebeta^alpha_i cdot mathbfB^i","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"where alpha_i depends on the atom type.","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"function calc_fitted_tot_energy(path, β, ncoeff, N1, N)\n    # Calculate y\n    lmp = LMP([\"-screen\",\"none\"])\n    bs = run_snap(lmp, path, rcut, twojmax)\n\n    E_tot_acc = 0.0\n    for n in 1:N1\n        E_atom_acc = β[1]\n        for k in 2:ncoeff\n            k2 = k - 1\n            E_atom_acc += β[k] * bs[k2, n]\n        end\n        E_tot_acc += E_atom_acc\n    end\n    for n in N1+1:N\n        E_atom_acc = β[ncoeff+1]\n        for k in ncoeff+2:2*ncoeff\n            k2 = k - ncoeff - 1\n            E_atom_acc += β[k] * bs[k2, n]\n        end\n        E_tot_acc += E_atom_acc\n    end\n    command(lmp, \"clear\")\n    return E_tot_acc\nend","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"Comparison between the potential energy calculated based on the DFT data, and the SNAP potential energy calculated based on the bispectrum components provided by snapjl and the fitted coefficients mathbfbeta.","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"@printf(\"Potential Energy, Fitted Potential Energy, Error (%%)\\n\")\nfor m = M1+1:M2\n    path = joinpath(DATA, string(m), \"DATA\")\n    rs = read_atomic_conf(m, N)\n    E_tot = calc_tot_energy(rcut, rs, N, ε_Ga_Ga, σ_Ga_Ga, ε_N_N, σ_N_N, A_Ga_N, ρ_Ga_N)\n    E_tot_fit = calc_fitted_tot_energy(path, β, ncoeff, N1, N)\n    @printf(\"%0.2f, %0.2f, %0.2f\\n\", E_tot, E_tot_fit,\n            abs(E_tot - E_tot_fit) / E_tot * 100.)\nend","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"","category":"page"},{"location":"generated/fitting_snap/","page":"Fitting SNAP","title":"Fitting SNAP","text":"This page was generated using Literate.jl.","category":"page"},{"location":"#LAMMPS.jl","page":"Home","title":"LAMMPS.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This packages wraps LAMMPS","category":"page"}]
}
