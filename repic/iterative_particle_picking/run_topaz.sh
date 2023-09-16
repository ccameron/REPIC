#!/usr/bin/env bash
#
#	run_topaz.sh - runs Topaz model on micrographs
#	author: Christopher JF Cameron & Sebastian JH Seager
#

#	track runtime
START=$(date +%s.%N)

#	collect environmental variables
if [ -z "${REPIC_MRC_DIR}" ]; then REPIC_MRC_DIR=0; fi
if [ -z "${REPIC_OUT_DIR}" ]; then REPIC_OUT_DIR=0; fi
if [ -z "${TOPAZ_ENV}" ]; then TOPAZ_ENV="topaz"; fi
if [ -z "${TOPAZ_SCALE}" ]; then TOPAZ_SCALE=0; fi
if [ -z "${TOPAZ_PARTICLE_RAD}" ]; then TOPAZ_PARTICLE_RAD=0; fi
if [ -z "${TOPAZ_MODEL}" ]; then :; fi

if [ -x "$(command -v micromamba)" ]; then
  eval "$(micromamba shell hook -s bash)" && micromamba activate ${TOPAZ_ENV}
elif [ -x "$(command -v conda)" ]; then
  eval "$(conda shell.bash hook)" && conda activate ${TOPAZ_ENV}
fi

#	identify particles
# if TOPAZ_MODEL is not set, use the general model (don't supply -m)
if [ -z "$TOPAZ_MODEL" ]; then
  topaz extract \
      -r ${TOPAZ_PARTICLE_RAD} \
      -x ${TOPAZ_SCALE} \
      -o ${REPIC_OUT_DIR}/predicted_particles_all_upsampled.txt \
      ${REPIC_MRC_DIR}/downsampled_mrc/*.mrc
else
  topaz extract \
      -r ${TOPAZ_PARTICLE_RAD} \
      -x ${TOPAZ_SCALE} \
      -m ${TOPAZ_MODEL} \
      -o ${REPIC_OUT_DIR}/predicted_particles_all_upsampled.txt \
      ${REPIC_MRC_DIR}/downsampled_mrc/*.mrc
fi

#	create empty BOX files to be replaced by /src/coord_converter.py
OUT_DIR=$(basename ${REPIC_MRC_DIR} | cut -d'_' -f1)
OUT_DIR=${REPIC_OUT_DIR}/BOX/${OUT_DIR}
mkdir -p ${OUT_DIR}
for file in ${REPIC_MRC_DIR}/downsampled_mrc/*.mrc; do
  base=$(basename ${file})
  base=${base/.mrc/.box}
  touch ${OUT_DIR}/${base}
done

#	save runtime to storage
END=$(date +%s.%N)
DIFF=$( echo "${END} - ${START}" | bc -l )
LABEL=$( basename ${REPIC_MRC_DIR} | sed -E 's/_[0-9]+//g' )
COUNT=$( ls ${REPIC_MRC_DIR}/*.mrc | wc -l )
echo -e """start\tend\tdifference\tN
${START}\t${END}\t${DIFF}\t${COUNT}""" > ${REPIC_OUT_DIR}/run_topaz_runtime_${LABEL}.tsv
