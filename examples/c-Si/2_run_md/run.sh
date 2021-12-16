lmp_mpi < in.run
if [ -d "data" ]
  then 
    rm -r data
fi
python gen_fitting_data.py
