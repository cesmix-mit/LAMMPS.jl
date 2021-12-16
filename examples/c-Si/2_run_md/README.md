# Crystalline silicon (c-Si) MD

This example involves a c-Si atoms vibrating at finite temperature.

First edit the following variables in `in.run` to have desired settings:

* `nsample` - declare however many timesteps we sample training data. 
* `nsteps` - number of timesteps

So the number of training configurations we generate will be `nsteps/nsample`.

Then run a MD simulation:

    lmp_mpi < in.run

You can view the results in VMD by doing `vmd dump.xyz` in the terminal. 

Then generate training data:

    python generate_fitting_data.py

which creates a directory `data` that houses directories containing:

* `DATA` - LAMMPS data file.
* `PE` - a file containing the potential energy of the configuration.
* `FORCES` - a file containing the forces on all atoms in the configuration.

To run everything together, simply run the shell script:

    chmod +x run.sh
    ./run.sh
