#!/usr/bin/env bash
#
#	get_examples.sh - retrieves example data for iterative esemble particle picking from AWS S3 bucket
#	author: Christopher JF Cameron
#
#	usage: bash src/get_examples.sh out_dir
#
#	positional arguments:
#		out_dir		path to output directory (will create if does not exist)
#

if [ -z "${1}" ]; then
    echo "Error - no output directory supplied"
    exit
fi

#	set output directory from command line argument
OUT_DIR=${1}

#	set up local directory
mkdir -p ${OUT_DIR}

#	iterate over filenames and download from AWS S3 bucket
files="Jul21_17_36_51 Jul21_17_39_03 Jul21_17_52_20 Jul21_17_56_42 Jul21_18_05_31 Jul21_18_38_48 Jul21_19_35_51 Jul21_19_38_03 Jul21_19_54_12 Jul21_19_56_25 Jul21_20_23_38 Jul21_20_39_19 Jul21_20_45_56 Jul21_20_50_20 Jul21_20_57_21 Jul21_21_24_01 Jul21_21_57_27 Jul21_22_04_08 Jul21_22_15_09 Jul21_22_37_22 Jul21_23_02_48 Jul21_23_05_02 Jul21_23_13_57 Jul21_23_16_09 Jul21_23_22_39 Jul21_23_24_50 Jul22_00_07_03 Jul22_00_13_45 Jul22_00_35_04 Jul22_00_37_23 Jul22_00_41_50 Jul22_00_52_53"

if which wget >/dev/null ; then
  for file in ${files}; do
    #	micrograph
    wget --no-check-certificate --no-proxy "http://org.gersteinlab.repic.s3.amazonaws.com/example_data_10057/${file}.mrc" -P ${OUT_DIR}
    #	normative particles
    wget --no-check-certificate --no-proxy "http://org.gersteinlab.repic.s3.amazonaws.com/example_data_10057/${file}.box" -P ${OUT_DIR}
  done
elif which curl >/dev/null ; then
  for file in ${files}; do
    curl http://org.gersteinlab.repic.s3.amazonaws.com/example_data_10057/${file}.mrc --insecure -o ${OUT_DIR}/${file}.mrc
    curl http://org.gersteinlab.repic.s3.amazonaws.com/example_data_10057/${file}.box --insecure -o ${OUT_DIR}/${file}.box
  done
else
    echo "Cannot download examples, neither wget nor curl is available."
fi
