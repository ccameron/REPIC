#!/usr/bin/env bash
#
#	fit_topaz.sh - fits Topaz model to micrographs
#	author: Christopher JF Cameron
#

#	track runtime
START=$(date +%s.%N)

#	collect environmental variables
if [ -z "${REPIC_TRAIN_MRC}" ]; then REPIC_TRAIN_MRC=0; fi
if [ -z "${REPIC_TRAIN_COORD}" ]; then REPIC_TRAIN_COORD=0; fi
if [ -z "${REPIC_VAL_MRC}" ]; then REPIC_VAL_MRC=0; fi
if [ -z "${REPIC_VAL_COORD}" ]; then REPIC_VAL_COORD=0; fi
if [ -z "${REPIC_NUM_PARTICLES}" ]; then REPIC_NUM_PARTICLES=0; fi
if [ -z "${REPIC_OUT_DIR}" ]; then REPIC_OUT_DIR=0; fi
if [ -z "${TOPAZ_ENV}" ]; then TOPAZ_ENV="topaz"; fi
if [ -z "${TOPAZ_SCALE}" ]; then TOPAZ_SCALE=0; fi
if [ -z "${TOPAZ_PARTICLE_RAD}" ]; then TOPAZ_PARTICLE_RAD=0; fi
if [ -z "${TOPAZ_BALANCE}" ]; then TOPAZ_BALANCE=0.0625; fi

eval "$(conda shell.bash hook)"
conda activate ${TOPAZ_ENV}

topaz convert -s ${TOPAZ_SCALE} \
    -o ${REPIC_OUT_DIR}/particles_train_downsampled.txt \
    ${REPIC_TRAIN_COORD}/*.box
topaz convert -s ${TOPAZ_SCALE} \
    -o ${REPIC_OUT_DIR}/particles_val_downsampled.txt \
    ${REPIC_VAL_COORD}/*.box

#	increase the number of expected particles <- suggestion by A. Noble at NCCAT-NYSB webinar
TOPAZ_NUM_PARTICLES=$(echo "$((${REPIC_NUM_PARTICLES} * 125/100))")

#	train Topaz model
topaz train -n ${TOPAZ_NUM_PARTICLES} \
    --num-workers 8 \
    --minibatch-size 64 \
    --minibatch-balance ${TOPAZ_BALANCE} \
    --train-images ${REPIC_TRAIN_MRC}/downsampled_mrc/ \
    --train-targets ${REPIC_OUT_DIR}/particles_train_downsampled.txt \
    --test-images ${REPIC_VAL_MRC}/downsampled_mrc/ \
    --test-targets ${REPIC_OUT_DIR}/particles_val_downsampled.txt \
    --save-prefix ${REPIC_OUT_DIR}/model \
    --no-pretrained \
    -o ${REPIC_OUT_DIR}/model_training.txt

rm -f ${REPIC_OUT_DIR}/particles_train_downsampled.txt
rm -f ${REPIC_OUT_DIR}/particles_val_downsampled.txt

conda deactivate

#	save runtime to storage
END=$(date +%s.%N)
DIFF=$( echo "${END} - ${START}" | bc -l )
TRAIN=$( ls ${REPIC_TRAIN_MRC}/*.mrc | wc -l )
VAL=$( ls ${REPIC_VAL_MRC}/*.mrc | wc -l )
echo -e """start\tend\tdifference\ttrain_N\tval_N
${START}\t${END}\t${DIFF}\t${TRAIN}\t${VAL}""" >  ${REPIC_OUT_DIR}/fit_topaz_runtime.tsv
