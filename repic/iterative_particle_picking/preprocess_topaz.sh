#!/usr/bin/env bash
#
#	preprocess_topaz.sh - downsamples micrographs for input to Topaz
#	author: Christopher JF Cameron
#

#	track runtime
START=$(date +%s.%N)

#	collect environmental variables
if [ -z "${REPIC_TRAIN_MRC}" ]; then REPIC_TRAIN_MRC=0; fi
if [ -z "${REPIC_VAL_MRC}" ]; then REPIC_VAL_MRC=0; fi
if [ -z "${REPIC_TEST_MRC}" ]; then REPIC_TES_MRC=0; fi
if [ -z "${REPIC_OUT_DIR}" ]; then REPIC_OUT_DIR=0; fi
if [ -z "${TOPAZ_ENV}" ]; then TOPAZ_ENV="topaz"; fi
if [ -z "${TOPAZ_SCALE}" ]; then TOPAZ_SCALE=0; fi

eval "$(conda shell.bash hook)"
conda activate ${TOPAZ_ENV}

#	downsample micrographs
topaz preprocess -s ${TOPAZ_SCALE} \
    --sample 1 \
    -o ${REPIC_TRAIN_MRC}/downsampled_mrc/ \
    ${REPIC_TRAIN_MRC}/*.mrc
topaz preprocess -s ${TOPAZ_SCALE} \
    --sample 1 \
    -o ${REPIC_VAL_MRC}/downsampled_mrc/ \
    ${REPIC_VAL_MRC}/*.mrc
topaz preprocess -s ${TOPAZ_SCALE} \
    --sample 1 \
    -o ${REPIC_TEST_MRC}/downsampled_mrc/ \
    ${REPIC_TEST_MRC}/*.mrc

conda deactivate

#	save runtime to storage
END=$(date +%s.%N)
DIFF=$( echo "${END} - ${START}" | bc -l )
TRAIN=$( ls ${REPIC_TRAIN_MRC}/*.mrc | wc -l )
VAL=$( ls ${REPIC_VAL_MRC}/*.mrc | wc -l )
TEST=$( ls ${REPIC_TEST_MRC}/*.mrc | wc -l )
echo -e """start\tend\tdifference\ttrain_N\tval_N\ttest_N
${START}\t${END}\t${DIFF}\t${TRAIN}\t${VAL}\t${TEST}""" > ${REPIC_OUT_DIR}/preprocess_topaz_runtime.txt
