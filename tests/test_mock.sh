#!/bin/bash

DIVE_CPP=~/OneDrive/Codes/DIVE_box/DIVE_box
PYDIVE=./dive.py
TEST_FILE=/home/daniel/scratch/projects/baosystematics/patchy_results/box5/redshift/nosyst/mocks_gal_xyz/CATALPTCICz0.638G960S1005638091_zspace.dat

#time $DIVE_CPP $TEST_FILE tests/voids_dive_box.dat 2500 0 999
#time $PYDIVE --input_catalog $TEST_FILE --output_catalog tests/voids_pydive.dat --box_size 2500 --rmin 0 --rmax 999

python tests/compare_distributions.py

