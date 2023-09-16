#!/usr/bin/env bash
#
#	run_cryolo.sh - runs SPHIRE-crYOLO model on micrographs
#	author: Christopher JF Cameron & Sebastian JH Seager
#

#	track runtime
START=$(date +%s.%N)

#	collect environmental variables
if [ -z "${REPIC_MRC_DIR}" ]; then REPIC_MRC_DIR=0; fi
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

# create SPHIRE-crYOLO config file
cryolo_gui.py config ${REPIC_OUT_DIR}/config_cryolo.json \
    ${REPIC_BOX_SIZE} \
    --filter LOWPASS \
    --low_pass_cutoff 0.1 \
    --filtered_output ${CRYOLO_FILTERED_DIR} \
    --log_path ${REPIC_OUT_DIR}/logs/

# identify particles
cryolo_predict.py -c ${REPIC_OUT_DIR}/config_cryolo.json \
    -w ${CRYOLO_MODEL} \
    -i ${REPIC_MRC_DIR} \
    -g 0 \
    -o ${REPIC_OUT_DIR} \
    -t 0.0 \
    --write_empty

#	save runtime to storage
END=$(date +%s.%N)
DIFF=$( echo "${END} - ${START}" | bc -l )
LABEL=$( basename ${REPIC_MRC_DIR} | sed -E 's/_[0-9]+//g' )
COUNT=$( ls ${REPIC_MRC_DIR}/*.mrc | wc -l )
echo -e """start\tend\tdifference\tN
${START}\t${END}\t${DIFF}\t${COUNT}""" > ${REPIC_OUT_DIR}/run_cryolo_runtime_${LABEL}.tsv
