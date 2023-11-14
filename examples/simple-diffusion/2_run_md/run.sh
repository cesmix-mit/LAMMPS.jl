lmp_mpi < in.run
rm -r data
python gen_fitting_data.py
julia fit_snap.jl
