# Crystalline silicon (c-Si) MD

This example involves a c-Si atoms vibrating at finite temperature.

First edit the following variables in `in.run` to have desired settings:

* `nsample` - declare however many timesteps we sample training data. 
* `nsteps` - number of timesteps

The number of training configurations we generate will be `nsteps/nsample`.

Then run a MD simulation:

    lmp_mpi < in.run

You can view the results in VMD by doing `vmd dump.xyz` in the terminal. 

Then generate training data:

    python generate_fitting_data_exyz.py

which creates `data.xyz` in extended xyz format.

To run everything together, simply run the shell script:

    chmod +x run.sh
    ./run.sh
