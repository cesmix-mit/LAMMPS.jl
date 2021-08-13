# Simple diffusion example for fitting SNAP.

This example involves a hydrogen atom diffusing in a palladium lattice (a situation commonly seen in hydrogen energy storage), modeled by the Morse potential. It's the most simple system of an atom diffusing in a lattice.

First edit the following variables in `in.run` to have desired settings:

* `nsample` - declare however many timesteps we sample training data. 
* `nsteps` - number of timesteps

So the number of training configurations we generate will be `nsteps/nsample`.

Then run a MD simulation of the diffusion process:

    lmp_mpi < in.run

You can view the results in VMD by doing `vmd dump.xyz` in the terminal. 

Then generate training data:

    python generate_fitting_data.py

which creates a directory `data` that houses directories containing:

* `DATA` - LAMMPS data file.
* `PE` - a file containing the potential energy of the configuration.
* `FORCES` - a file containing the forces on all atoms in the configuration.

To train a SNAP potential on this data, edit `fit_snap.jl` to have the correct settings. The following variables are relevant:

* `M` - number of configurations to train to.

To run everything together, simply run the shell script:

    chmod +x run.sh
    ./run.sh
