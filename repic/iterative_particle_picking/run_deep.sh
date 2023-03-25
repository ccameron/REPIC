#!/usr/bin/env bash
#
#	run_deep.sh - runs DeepPicker model on micrographs
#	author: Christopher JF Cameron & Sebastian JH Seager
#

#	track runtime
START=$(date +%s.%N)

#	collect environmental variables
if [ -z "${REPIC_MRC_DIR}" ]; then REPIC_MRC_DIR=0; fi
if [ -z "${REPIC_BOX_SIZE}" ]; then REPIC_BOX_SIZE=0; fi
if [ -z "${REPIC_OUT_DIR}" ]; then REPIC_OUT_DIR=0; fi
if [ -z "${REPIC_UTILS}" ]; then REPIC_UTILS=0; fi
if [ -z "${DEEP_ENV}" ]; then DEEP_ENV="deep"; fi
if [ -z "${DEEP_DIR}" ]; then DEEP_DIR="./DeepPicker-python"; fi
if [ -z "${DEEP_MODEL}" ]; then DEEP_MODEL=${DEEP_DIR}/trained_model/model_demo_type3; fi

eval "$(conda shell.bash hook)"
conda activate ${DEEP_ENV}

python ${DEEP_DIR}/autoPick.py \
    --inputDir ${REPIC_MRC_DIR}/ \
    --pre_trained_model ${DEEP_MODEL} \
    --particle_size ${REPIC_BOX_SIZE} \
    --outputDir ${REPIC_OUT_DIR}/STAR \
    --coordinate_symbol _deeppicker \
    --threshold 0.0

conda deactivate

#	save runtime to storage
END=$(date +%s.%N)
DIFF=$( echo "${END} - ${START}" | bc -l )
LABEL=$( basename ${REPIC_MRC_DIR} | sed -E 's/_[0-9]+//g' )
COUNT=$( ls ${REPIC_MRC_DIR}/*.mrc | wc -l )
echo -e """start\tend\tdifference\tN
${START}\t${END}\t${DIFF}\t${COUNT}""" > ${REPIC_OUT_DIR}/run_deep_runtime_${LABEL}.tsv
