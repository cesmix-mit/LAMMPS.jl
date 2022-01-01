"""
Generate fitting data based on LAMMPS simulation output dump.forces_and_positions.py
"""

import os
import numpy as np

# Declare quantities
# first index is configuration, and other indices are atoms.
# e.g. positions[m][n][a] is Cartesian coordinate a of atom n in config m.
# e.g. energies[m] is energy of config m
# stresses[m] are the 6 symmetric components of the stress tensor for config m, ordered like xx yy zz yz xz xy (see in.run for more details).
positions = []
forces= []
species = []
natoms = []
boxes = []
energies = []
stresses = []

# Gather all per-atom quantities
config_count = 0
with open('dump.positions_and_forces') as fh:
  for line in fh:
      if "ITEM: NUMBER OF ATOMS" in line:
        config_count = config_count+1
        line = fh.readline()
        natoms_config = int([int(x) for x in line.split()][0])
        #print(natoms_config)
        line = fh.readline()
        # Read box
        box_x = [float(x) for x in fh.readline().split()]
        box_y = [float(x) for x in fh.readline().split()]
        box_z = [float(x) for x in fh.readline().split()]
        # Skip next line
        line = fh.readline()
        # Read atom quantities
        x_config = []
        f_config = []
        species_config = []
        types_config = []
        for i in range(0,natoms_config):
          line = fh.readline()
          line_split = line.split()
          tag = int(line_split[0])
          typ = int(line_split[1])
          xcoor = float(line_split[2])
          ycoor = float(line_split[3])
          zcoor = float(line_split[4])
          fx = float(line_split[5])
          fy = float(line_split[6])
          fz = float(line_split[7])
          x_config.append([xcoor,ycoor,zcoor])
          f_config.append([fx,fy,fz])
          species_config.append("Si")
          types_config.append(typ)
          #fh_w.write("%d %d %f %f %f\n" % (tag,typ, x,y,z))
          #fh_f.write("%f\n%f\n%f\n" % (fx,fy,fz))
          
        # Append to total quantity array
        positions.append(x_config)
        forces.append(f_config)
        species.append(species_config)
        natoms.append(natoms_config)
        boxes.append([box_x[1],box_y[1],box_z[1]])
        

# Read energies and stresses
#config_count = 0
with open('log.lammps') as fh:
  for line in fh:
      if "Step PotEng Temp" in line:
        while("Loop time" not in line):
          line = fh.readline()
          #print(line)
          #config_count = config_count+1
          if ("Loop time" in line):
            break
          else:
            line_split = [float(x) for x in line.split()]
            #print(line_split)
            pe_config = line_split[1]
            stress_config = [line_split[3],line_split[4],line_split[5],line_split[6],line_split[7],line_split[8]]
  
            
            # Append to total quantity array
            energies.append(pe_config)
            stresses.append(stress_config)

# Convert to np arrays
positions = np.array(positions)
forces = np.array(forces)
species = np.array(species)
energies = np.array(energies)
stresses = np.array(stresses)
natoms = np.array(natoms)
boxes = np.array(boxes)
#print(stresses)

# Write EXYZ file
fh = open("data.xyz", 'w')
for m in range(0,config_count):
  fh.write("%d\n" % (natoms[m]) )
  line = 'Lattice="%e 0.0 0.0 0.0 %e 0.0 0.0 0.0 %e" Properties=species:S:1:pos:r:3:forces:r:3 energy=%e stress="%e %e %e %e %e %e" pbc="T T T"\n' % (boxes[m][0],boxes[m][1],boxes[m][2],energies[m], stresses[m][0],stresses[m][1],stresses[m][2],stresses[m][3],stresses[m][4],stresses[m][5])
  #print(line)
  fh.write(line)
  for n in range(0,natoms[m]):
    x = positions[m][n][0]
    y = positions[m][n][1]
    z = positions[m][n][2]
    fx = forces[m][n][0]
    fy = forces[m][n][1]
    fz = forces[m][n][2]
    line = "%s %e %e %e %e %e %e\n" % (species[m][n],x,y,z,fx,fy,fz)
    fh.write(line)
fh.close()

