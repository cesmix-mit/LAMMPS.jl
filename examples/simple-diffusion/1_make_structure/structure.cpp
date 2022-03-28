#include <iostream>
#include <cmath>

int main() {

  double a = 3.0;
  int lx = 2;
  int ly = 2;
  int lz = 2;
  int nunit = 8; // number of atoms in unit cell
  int nvals = 200;
  int nbatoms = 1; // number basis atoms

  double amat[3][3];
  amat[0][0] = a*0.5;
  amat[0][1] = a*0.0;
  amat[0][2] = a*0.0;
  amat[1][0] = a*0.0;
  amat[1][1] = a*0.5;
  amat[1][2] = a*0.0;
  amat[2][0] = a*0.0;
  amat[2][1] = a*0.0;
  amat[2][2] = a*0.5;

  double basis[1][3];
  basis[0][0] = 0.0;
  basis[0][1] = 0.0;
  basis[0][2] = 0.0;


  double box[3];
  box[0] = lx*a;
  box[1] = ly*a;
  box[2] = lz*a;
  // Round to 6 decimals
  box[0] = std::ceil(box[0] * 1e6) / 1e6;
  box[1] = std::ceil(box[1] * 1e6) / 1e6;
  box[2] = std::ceil(box[2] * 1e6) / 1e6;

  printf("Box: %f %f %f\n", box[0],box[1],box[2]);
  FILE * fh_box;
  fh_box = fopen("BOX", "w");
  fprintf(fh_box, "%f %f %f\n", box[0], box[1], box[2]);
  fclose(fh_box); 

  int supposed_atoms = lx*ly*lz*nunit*nbatoms;

  printf("%d supposed atoms.\n", supposed_atoms+1); // Add 1 to supposed_atoms because we add a diffusion atom later. 
  FILE * fh_xyz;
  //FILE * fh_dat; 
  fh_xyz = fopen("structure.xyz", "w");
  //fh_dat = fopen("DATA", "w");
  fprintf(fh_xyz, "%d\n", supposed_atoms+1); // Add 1 to supposed_atoms because we add a diffusion atom later. 
  fprintf(fh_xyz, "Comment line\n"); 

  double x,y,z;
  int count = 0;
  int type;

  // Place an atom for diffusion
  fprintf(fh_xyz, "%d 2 %f %f %f\n", count+1,a/4,a/4,a/4);
  count++;

  for (int b=0; b<nbatoms; b++){
    for (int n1=-1.0*nvals; n1<nvals; n1++){
      for (int n2=-1.0*nvals; n2<nvals; n2++){
        for (int n3=-1.0*nvals; n3<nvals; n3++){
          x = n1*amat[0][0] + n2*amat[1][0] + n3*amat[2][0] + basis[b][0];
          y = n1*amat[0][1] + n2*amat[1][1] + n3*amat[2][1] + basis[b][1];
          z = n1*amat[0][2] + n2*amat[1][2] + n3*amat[2][2] + basis[b][2];

          // round to 6 decimals
          x = std::ceil(x * 1e6) / 1e6;
          y = std::ceil(y * 1e6) / 1e6;
          z = std::ceil(z * 1e6) / 1e6;

          if ( (0<=x) && (x < box[0]) && (0<=y) && (y < box[1]) && (0<=z) && (z < box[2]) ){

            type=1;
            //fprintf(fh_xyz, "%d %d %f %f %f\n", count+1,type,x,y,z);
            fprintf(fh_xyz, "%d %d %f %f %f\n", count+1,type,x,y,z);
            count++;
          }

        }

      }

    }

  } 
  
  printf("Made %d atoms.\n", count);
  /*
  if (count != supposed_atoms){
    printf("DID NOT MAKE CORRECT NUMBER OF ATOMS!\n"); 
  } 
  */

  fclose(fh_xyz);
  //fclose(fh_dat);
  

  return 0;

}


