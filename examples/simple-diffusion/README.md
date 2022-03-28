# Simple diffusion example for fitting SNAP.

This example involves a hydrogen atom diffusing in a palladium lattice (a situation commonly seen in hydrogen energy storage), modeled by the Morse potential. It's the most simple system of an atom diffusing in a lattice.

`1_make_structure` houses a C++ script for making the structure of the system.
`2_run_md` houses LAMMPS input scripts to run MD, save the training data, and then fit SNAP to the training data.
`3_run_md_snap` houses LAMMPS input scripts to run MD with the fitted SNAP potential, in LAMMPS.

Currently this example produces an unstable SNAP potential, because we need to fit to forces instead of just energy.
