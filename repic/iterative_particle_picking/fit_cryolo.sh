#!/usr/bin/env bash
#
#	fit_cryolo.sh - fits SPHIRE-crYOLO  model to micrographs
#	author: Christopher JF Cameron
#

#	track runtime
START=$(date +%s.%N)

#	collect environmental variables
if [ -z "${REPIC_TRAIN_MRC}" ]; then REPIC_TRAIN_MRC=0; fi
if [ -z "${REPIC_TRAIN_COORD}" ]; then REPIC_TRAIN_COORD=0; fi
if [ -z "${REPIC_VAL_MRC}" ]; then REPIC_VAL_MRC=0; fi
if [ -z "${REPIC_VAL_COORD}" ]; then REPIC_VAL_COORD=0; fi
if [ -z "${REPIC_BOX_SIZE}" ]; then REPIC_BOX_SIZE=0; fi
if [ -z "${REPIC_OUT_DIR}" ]; then REPIC_OUT_DIR=0; fi
if [ -z "${REPIC_UTILS}" ]; then REPIC_UTILS=0; fi
if [ -z "${CRYOLO_ENV}" ]; then CRYOLO_ENV="cryolo"; fi
if [ -z "${CRYOLO_FILTERED_DIR}" ]; then CRYOLO_FILTERED_DIR=${REPIC_OUT_DIR}/filtered_tmp/; fi

if [ -x "$(command -v micromamba)" ]; then
  eval "$(micromamba shell hook -s bash)" && micromamba activate ${CRYOLO_ENV}
elif [ -x "$(command -v conda)" ]; then
  eval "$(conda shell.bash hook)" && conda activate ${CRYOLO_ENV}
fi

#	create SPHIRE-crYOLO config file
#	batch_size set to 2 for consistency between development machines
cryolo_gui.py config ${REPIC_OUT_DIR}/config_cryolo.json \
    ${REPIC_BOX_SIZE} \
    --train_image_folder ${REPIC_TRAIN_MRC} \
    --train_annot_folder ${REPIC_TRAIN_COORD} \
    --valid_image_folder ${REPIC_VAL_MRC} \
    --valid_annot_folder ${REPIC_VAL_COORD}\
    --saved_weights_name ${REPIC_OUT_DIR}/learned_weights.h5 \
    --filter LOWPASS \
    --low_pass_cutoff 0.1 \
    --batch_size 2 \
    --filtered_output ${CRYOLO_FILTERED_DIR} \
    --log_path ${REPIC_OUT_DIR}/logs/

#	fit SPHIRE-crYOLO model
cryolo_train.py -c ${REPIC_OUT_DIR}/config_cryolo.json \
    -w 5 \
    -g 0 \
    -e 32 \
    --seed 1 \

#	save runtime to storage
END=$(date +%s.%N)
DIFF=$( echo "${END} - ${START}" | bc -l )
TRAIN=$( ls ${REPIC_TRAIN_MRC}/*.mrc | wc -l )
VAL=$( ls ${REPIC_VAL_MRC}/*.mrc | wc -l )
echo -e """start\tend\tdifference\ttrain_N\tval_N
${START}\t${END}\t${DIFF}\t${TRAIN}\t${VAL}""" >  ${REPIC_OUT_DIR}/fit_cryolo_runtime.tsv
