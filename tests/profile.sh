#!/bin/bash

DIVE_BOX_CPP=~/codes/DIVE_box/DIVE_box
DIVE_CPP=~/codes/DIVE/DIVE
RDZ2XYZ=~/codes/rdz2xyz/rdz2xyz
PYDIVE='python -m cProfile -s time pydive/dive.py'
TEST_BOX=tests/CATALPTCICz0.638G960S1005638091_zspace.dat
DIVE_BOX=CATALPTCICz0.638G960S1005638091_zspace.VOID.dat
TEST_LC=tests/galaxy_DR12v5_CMASSLOWZTOT_South.dat.ascii
TEST_LC_XYZ=tests/galaxy_DR12v5_CMASSLOWZTOT_South.dat.XYZ.ascii
touch tests/time_dive_box.dat
touch tests/time_pydive_box.dat
touch tests/time_dive_lc.dat
touch tests/time_pydive_lc.dat
#TEST_FILE=tests/points.dat
echo "Testing DIVE on a box"
#time srun -n1 -c1  $DIVE_BOX_CPP $TEST_BOX tests/voids_dive_box.dat 2500 0 999 #2> tests/time_dive_box.dat

echo "Testing PyDIVE on a box"
#time srun -n1 -c32 $PYDIVE --input_catalog $TEST_BOX --output_catalog tests/voids_pydive.dat --box_size 2500 --rmin 0 --rmax 999 --is-box  #2> tests/time_pydive_box.dat

echo "Testing DIVE on a light cone"
#rm -v tests/galaxy_DR12v5_CMASSLOWZTOT_South.dat.XYZ.ascii
#time srun -n1 -c1 $RDZ2XYZ -c tests/rdz2xyz.conf
#time srun -n1 -c1 $DIVE_CPP $TEST_LC_XYZ tests/voids_dive_lc.dat   #2> tests/time_dive_lc.dat

echo "Profiling PyDIVE on a light cone"
time srun -n1 -c32 $PYDIVE --input_catalog $TEST_LC --output_catalog tests/voids_pydive_lc.dat --rmin 0 --rmax 9999999999  -c > tests/profile_lc_parallel.dat
time srun -n1 -c32 $PYDIVE --input_catalog $TEST_LC --output_catalog tests/voids_pydive_lc.dat --rmin 0 --rmax 9999999999  -c -n 1 > tests/profile_lc.dat
 
#python tests/compare_distributions.py | tee  tests/test_statistics_results.dat

