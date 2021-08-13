"""
Generate fitting data based on LAMMPS simulation output dump.forces_and_positions.py
"""

import os

def write_data_intro(fh_w, natoms, box_x,box_y,box_z):
  xlo = box_x[0]
  xhi = box_x[1]
  ylo = box_y[0]
  yhi = box_y[1]
  zlo = box_z[0]
  zhi = box_z[1]
  fh_w.write("LAMMPS DATA file\n")
  fh_w.write("\n")
  fh_w.write("%d atoms\n" % (natoms))
  fh_w.write("\n")
  fh_w.write("2 atom types\n")
  fh_w.write("\n")
  fh_w.write("%f %f xlo xhi\n" % (xlo,xhi))
  fh_w.write("%f %f ylo yhi\n" % (ylo,yhi))
  fh_w.write("%f %f zlo zhi\n" % (zlo,zhi))
  fh_w.write("\n")
  fh_w.write("Masses\n")
  fh_w.write("\n")
  fh_w.write("1 106.42\n")
  fh_w.write("2 2.0\n")
  fh_w.write("\n")
  fh_w.write("Atoms\n")
  fh_w.write("\n")

#os.system("rm -r data")
os.mkdir("data")
config_count = 0
with open('dump.positions_and_forces') as fh:
  for line in fh:
      if "ITEM: NUMBER OF ATOMS" in line:
        config_count = config_count+1
        line = fh.readline()
        natoms = int([int(x) for x in line.split()][0])
        #print(natoms)
        line = fh.readline()
        # Read box
        box_x = [float(x) for x in fh.readline().split()]
        box_y = [float(x) for x in fh.readline().split()]
        box_z = [float(x) for x in fh.readline().split()]
        # Skip next line
        line = fh.readline()
        # Make DATA file
        os.mkdir("data/%d" % (config_count))
        fh_w = open("data/%d/DATA" % (config_count), 'w')
        fh_f = open("data/%d/FORCES" % (config_count), 'w')
        write_data_intro(fh_w, natoms, box_x, box_y, box_z)
        # Read atom quantities
        for i in range(0,natoms):
          line = fh.readline()
          line_split = line.split()
          tag = int(line_split[0])
          typ = int(line_split[1])
          x = float(line_split[2])
          y = float(line_split[3])
          z = float(line_split[4])
          fx = float(line_split[5])
          fy = float(line_split[6])
          fz = float(line_split[7])
          fh_w.write("%d %d %f %f %f\n" % (tag,typ, x,y,z))
          fh_f.write("%f\n%f\n%f\n" % (fx,fy,fz))
        fh_w.close()
        fh_f.close()

# Read energies
config_count = 0
with open('log.lammps') as fh:
  for line in fh:
      if "Step TotEng PotEng Temp" in line:
        while("Loop time" not in line):
          line = fh.readline()
          #print(line)
          config_count = config_count+1
          if ("Loop time" in line):
            break
          else:
            pe = [float(x) for x in line.split()][2]
            #print(pe)
            fh_w = open("data/%d/PE" % (config_count), 'w')
            fh_w.write("%.10e\n" % (pe))
            fh_w.close()


