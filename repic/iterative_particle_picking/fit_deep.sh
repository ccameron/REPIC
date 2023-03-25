#!/usr/bin/env bash
#
#	fit_deep.sh - fits DeepPicker model to micrographs
# author: Christopher JF Cameron
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
if [ -z "${DEEP_ENV}" ]; then DEEP_ENV="deep"; fi
if [ -z "${DEEP_DIR}" ]; then DEEP_DIR="./DeepPicker-python"; fi
if [ -z "${DEEP_BATCH_SIZE}" ]; then DEEP_BATCH_SIZE=32; fi

#	convert train and val particles to STAR format for input to DeepPicker
for BOX_FILE in ${REPIC_TRAIN_COORD}/*.box; do
  DIRNAME=$(dirname ${BOX_FILE})
  python ${REPIC_UTILS}/coord_converter.py ${BOX_FILE} ${DIRNAME}/STAR -f box -t star -b ${REPIC_BOX_SIZE} --header --force
done
REPIC_TRAIN_COORD=${REPIC_TRAIN_COORD}/STAR
for BOX_FILE in ${REPIC_VAL_COORD}/*.box; do
  DIRNAME=$(dirname ${BOX_FILE})
  python ${REPIC_UTILS}/coord_converter.py ${BOX_FILE} ${DIRNAME}/STAR -f box -t star -b ${REPIC_BOX_SIZE} --header --force
done
REPIC_VAL_COORD=${REPIC_VAL_COORD}/STAR

eval "$(conda shell.bash hook)"
conda deactivate

#	softlink micrographs to coordinate directory
cp -s ${REPIC_TRAIN_MRC}/*.mrc ${REPIC_TRAIN_COORD}/
cp -s ${REPIC_VAL_MRC}/*.mrc ${REPIC_VAL_COORD}/

conda activate ${DEEP_ENV}

python ${DEEP_DIR}/train.py --train_type 1 \
    --train_inputDir ${REPIC_TRAIN_COORD} \
    --validation_inputDir ${REPIC_VAL_COORD} \
    --particle_size ${REPIC_BOX_SIZE} \
    --coordinate_symbol '' \
    --model_retrain \
    --model_load_file ${DEEP_DIR}/trained_model/model_demo_type3 \
    --model_save_dir ${REPIC_OUT_DIR}/ \
    --model_save_file model_demo_type3_refined \
    --batch_size ${DEEP_BATCH_SIZE}

#	remove softlinks
rm -rf ${REPIC_TRAIN_COORD}
rm -rf ${REPIC_VAL_COORD}

conda deactivate

#	save runtime to storage
END=$(date +%s.%N)
DIFF=$( echo "${END} - ${START}" | bc -l )
TRAIN=$( ls ${REPIC_TRAIN_MRC}/*.mrc | wc -l )
VAL=$( ls ${REPIC_VAL_MRC}/*.mrc | wc -l )
echo -e """start\tend\tdifference\ttrain_N\tval_N
${START}\t${END}\t${DIFF}\t${TRAIN}\t${VAL}""" >  ${REPIC_OUT_DIR}/fit_deep_runtime.tsv
