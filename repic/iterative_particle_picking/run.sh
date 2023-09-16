#!/usr/bin/env bash
#
#	run.sh - perform particle picking using iterative ensemble learning
#	author: Christopher JF Cameron
#

#  set filepath to REPIC install src/
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPIC=$( dirname ${SCRIPT_DIR} )
export REPIC_UTILS=${REPIC}/utils

#  parse command line arguments
IN_DIR=${1}
[ ! -d ${IN_DIR} ] && echo "Error - input directory '${IN_DIR}' does not exist."
IN_DIR=$( cd ${IN_DIR} && pwd )	#	set absolute path
EMPIAR_ID=$( basename ${IN_DIR} )
#  number of iterations to perform (+1 for initial round w pre-trained models)
NUM_ROUNDS=${2}
export REPIC_BOX_SIZE=${3}
export REPIC_NUM_PARTICLES=${4}
LABEL=${5}
export REPIC_TRAIN_MRC=${IN_DIR}/iterative_particle_picking/train/${LABEL}
if [ "auto" = ${6} ]; then USE_MANUAL_LABELS=false; else USE_MANUAL_LABELS=true;fi
if [[ "${7}" == 1 ]]; then GET_SCORE=true; else GET_SCORE=false; fi
export REPIC_VAL_MRC=${IN_DIR}/iterative_particle_picking/val/
export REPIC_TEST_MRC=${IN_DIR}/iterative_particle_picking/test
REPIC_COORD=${IN_DIR}/iterative_particle_picking
export CRYOLO_ENV=${8}
export CRYOLO_MODEL=${9}
export DEEP_ENV=${10}
export DEEP_DIR=${11}
export TOPAZ_ENV=${12}
export TOPAZ_SCALE=${13}
export TOPAZ_PARTICLE_RAD=${14}

#  CrYOLO filtered micrograph directory
export CRYOLO_FILTERED_DIR=${IN_DIR}/iterative_particle_picking/cryolo_filtered_tmp

###
#  step 1 - create train/val/test data sets based on micrograph defocus values
###

#	clean up iterative_particle_picking/
rm -rf ${IN_DIR}/iterative_particle_picking/{round_*,train,val,test,cryolo_filtered_tmp,iteration_plots,preprocess_topaz*}
mkdir -p ${IN_DIR}/iterative_particle_picking

echo "Building train/val/test subsets ... "
python ${REPIC_UTILS}/build_subsets.py ${IN_DIR}/data/defocus_${EMPIAR_ID}.txt ${IN_DIR}/data ${IN_DIR}/data ${IN_DIR}/iterative_particle_picking --train_set ${LABEL}
if [ $? != 0 ]; then	#	end script prematurely if error encountered in by repic build_subsets
  exit
fi
# multiple train sets created when possible (depends on number of micrographs)

echo "Downsampling micrographs for input to Topaz ... "
export REPIC_OUT_DIR=${IN_DIR}/iterative_particle_picking/
bash ${REPIC}/iterative_particle_picking/preprocess_topaz.sh &> ${REPIC_OUT_DIR}/preprocess_topaz.log

SUB_DIR=${IN_DIR}/iterative_particle_picking/round_0
mkdir -p ${SUB_DIR}
if ! ${USE_MANUAL_LABELS}; then

  ##
  #  step 2 - apply general models to train and test sets
  ##

  echo -e """\n--- Identifying particles in '${LABEL}' ---
  Applying general models to train, val, and test micrographs ...
  --- Round: 0 ---
  \tSPHIRE-crYOLO ... "
  # SPHIRE-crYOLO
  # train set prediction
  export REPIC_MRC_DIR=${REPIC_TRAIN_MRC}
  export REPIC_OUT_DIR=${SUB_DIR}/${LABEL}/cryolo
  mkdir -p ${REPIC_OUT_DIR}/BOX/{train,val,test}
  rm -rf ${REPIC_OUT_DIR}/{CBOX,STAR}/*
  bash ${REPIC}/iterative_particle_picking/run_cryolo.sh &> ${REPIC_OUT_DIR}/iter_train.log
  python ${REPIC_UTILS}/coord_converter.py ${REPIC_OUT_DIR}/CBOX/*.cbox ${REPIC_OUT_DIR}/BOX/train/ -f cbox -t box -b ${REPIC_BOX_SIZE} --round 0 --force &> ${REPIC_OUT_DIR}/convert_train.log
  #  val set prediction
  export REPIC_MRC_DIR=${REPIC_VAL_MRC}
  rm -rf ${REPIC_OUT_DIR}/{CBOX,STAR}/*
  bash ${REPIC}/iterative_particle_picking/run_cryolo.sh &> ${REPIC_OUT_DIR}/iter_val.log
  python ${REPIC_UTILS}/coord_converter.py ${REPIC_OUT_DIR}/CBOX/*.cbox ${REPIC_OUT_DIR}/BOX/val/ -f cbox -t box -b ${REPIC_BOX_SIZE} --round 0 --force &> ${REPIC_OUT_DIR}/convert_val.log
  #  test set prediction
  export REPIC_MRC_DIR=${REPIC_TEST_MRC}
  rm -rf ${REPIC_OUT_DIR}/{CBOX,STAR}/*
  bash ${REPIC}/iterative_particle_picking/run_cryolo.sh &> ${REPIC_OUT_DIR}/iter_test.log
  python ${REPIC_UTILS}/coord_converter.py ${REPIC_OUT_DIR}/CBOX/*.cbox ${REPIC_OUT_DIR}/BOX/test/ -f cbox -t box -b ${REPIC_BOX_SIZE} --round 0 --force &> ${REPIC_OUT_DIR}/convert_test.log
  if ${GET_SCORE}; then
    python ${REPIC_UTILS}/score_detections.py -g ${REPIC_COORD}/train/${LABEL}/*.box -p ${REPIC_OUT_DIR}/BOX/train/*.box -c 0.3 &> ${REPIC_OUT_DIR}/score_train.log
    python ${REPIC_UTILS}/score_detections.py -g ${REPIC_COORD}/val/*.box -p ${REPIC_OUT_DIR}/BOX/val/*.box -c 0.3 &> ${REPIC_OUT_DIR}/score_val.log
    python ${REPIC_UTILS}/score_detections.py -g ${REPIC_COORD}/test/*.box -p ${REPIC_OUT_DIR}/BOX/test/*.box -c 0.3 &> ${REPIC_OUT_DIR}/score_test.log
  fi

  echo -e "\tDeepPicker ... "
  #	DeepPicker
  export REPIC_MRC_DIR=${REPIC_TRAIN_MRC}
  export REPIC_OUT_DIR=${SUB_DIR}/${LABEL}/deep
  mkdir -p ${REPIC_OUT_DIR}/BOX/{train,val,test}
  #	train set prediction
  rm -rf ${REPIC_OUT_DIR}/STAR/*.star
  bash ${REPIC}/iterative_particle_picking/run_deep.sh &> ${REPIC_OUT_DIR}/iter_train.log
  python ${REPIC_UTILS}/coord_converter.py ${REPIC_OUT_DIR}/STAR/*.star ${REPIC_OUT_DIR}/BOX/train/ -f star -t box -b ${REPIC_BOX_SIZE} --round 0 --force &> ${REPIC_OUT_DIR}/convert_train.log
  #  val set prediction
  export REPIC_MRC_DIR=${REPIC_VAL_MRC}
  rm -rf ${REPIC_OUT_DIR}/STAR/*.star
  bash ${REPIC}/iterative_particle_picking/run_deep.sh &> ${REPIC_OUT_DIR}/iter_val.log
  python ${REPIC_UTILS}/coord_converter.py ${REPIC_OUT_DIR}/STAR/*.star ${REPIC_OUT_DIR}/BOX/val/ -f star -t box -b ${REPIC_BOX_SIZE} --round 0 --force &> ${REPIC_OUT_DIR}/convert_val.log
  #  test set prediction
  export REPIC_MRC_DIR=${REPIC_TEST_MRC}
  rm -rf ${REPIC_OUT_DIR}/STAR/*.star
  bash ${REPIC}/iterative_particle_picking/run_deep.sh &> ${REPIC_OUT_DIR}/iter_test.log
  python ${REPIC_UTILS}/coord_converter.py ${REPIC_OUT_DIR}/STAR/*.star ${REPIC_OUT_DIR}/BOX/test/ -f star -t box -b ${REPIC_BOX_SIZE} --round 0 --force &> ${REPIC_OUT_DIR}/convert_test.log
  if ${GET_SCORE}; then
    python ${REPIC_UTILS}/score_detections.py -g ${REPIC_COORD}/train/${LABEL}/*.box -p ${REPIC_OUT_DIR}/BOX/train/*.box -c 0.5 &> ${REPIC_OUT_DIR}/score_train.log
    python ${REPIC_UTILS}/score_detections.py -g ${REPIC_COORD}/val/*.box -p ${REPIC_OUT_DIR}/BOX/val/*.box -c 0.5 &> ${REPIC_OUT_DIR}/score_val.log
    python ${REPIC_UTILS}/score_detections.py -g ${REPIC_COORD}/test/*.box -p ${REPIC_OUT_DIR}/BOX/test/*.box -c 0.5 &> ${REPIC_OUT_DIR}/score_test.log
  fi

  echo -e "\tTopaz ... "
  # Topaz
  export REPIC_MRC_DIR=${REPIC_TRAIN_MRC}
  export REPIC_OUT_DIR=${SUB_DIR}/${LABEL}/topaz
  mkdir -p ${REPIC_OUT_DIR}/BOX/{train,val,test}
  # train set prediction
  rm -rf ${REPIC_OUT_DIR}/downsampled_mrc ${REPIC_OUT_DIR}/predicted_particles_all_upsampled.txt
  bash ${REPIC}/iterative_particle_picking/run_topaz.sh &> ${REPIC_OUT_DIR}/iter_train.log
  python ${REPIC_UTILS}/coord_converter.py ${REPIC_OUT_DIR}/predicted_particles_all_upsampled.txt ${REPIC_OUT_DIR}/BOX/train -f tsv -t box -b ${REPIC_BOX_SIZE} -c 1 2 none none 3 0 --header --multi_out --round 0 --force &> ${REPIC_OUT_DIR}/convert_train.log
  #  val set prediction
  export REPIC_MRC_DIR=${REPIC_VAL_MRC}
  rm -rf ${REPIC_OUT_DIR}/downsampled_mrc ${REPIC_OUT_DIR}/predicted_particles_all_upsampled.txt
  bash ${REPIC}/iterative_particle_picking/run_topaz.sh &> ${REPIC_OUT_DIR}/iter_val.log
  python ${REPIC_UTILS}/coord_converter.py ${REPIC_OUT_DIR}/predicted_particles_all_upsampled.txt ${REPIC_OUT_DIR}/BOX/val -f tsv -t box -b ${REPIC_BOX_SIZE} -c 1 2 none none 3 0 --header --multi_out --round 0 --force &> ${REPIC_OUT_DIR}/convert_val.log
  # test set prediction
  export REPIC_MRC_DIR=${REPIC_TEST_MRC}
  rm -rf ${REPIC_OUT_DIR}/downsampled_mrc ${REPIC_OUT_DIR}/predicted_particles_all_upsampled.txt
  bash ${REPIC}/iterative_particle_picking/run_topaz.sh &> ${REPIC_OUT_DIR}/iter_test.log
  python ${REPIC_UTILS}/coord_converter.py ${REPIC_OUT_DIR}/predicted_particles_all_upsampled.txt ${REPIC_OUT_DIR}/BOX/test -f tsv -t box -b ${REPIC_BOX_SIZE} -c 1 2 none none 3 0 --header --multi_out --round 0 --force &> ${REPIC_OUT_DIR}/convert_test.log
  if ${GET_SCORE}; then
  #	particle probability of >=0.5 is equal to log-likelihood ratio of >=0.0. See (line 17): https://github.com/tbepler/topaz/blob/master/tutorial/02_walkthrough.ipynb
    python ${REPIC_UTILS}/score_detections.py -g ${REPIC_COORD}/train/${LABEL}/*.box -p ${REPIC_OUT_DIR}/BOX/train/*.box -c 0. &> ${REPIC_OUT_DIR}/score_train.log
    python ${REPIC_UTILS}/score_detections.py -g ${REPIC_COORD}/val/*.box -p ${REPIC_OUT_DIR}/BOX/val/*.box -c 0. &> ${REPIC_OUT_DIR}/score_val.log
    python ${REPIC_UTILS}/score_detections.py -g ${REPIC_COORD}/test/*.box -p ${REPIC_OUT_DIR}/BOX/test/*.box -c 0. &> ${REPIC_OUT_DIR}/score_test.log
  fi

  echo -e "\tBuilding consensus ... "
  TMP=${SUB_DIR}/${LABEL}/tmp
  rm -rf ${TMP}
  REPIC_OUT_DIR=${SUB_DIR}/${LABEL}/clique_files
  mkdir -p ${REPIC_OUT_DIR}
  # train consensus
  mkdir -p ${TMP}/{crYOLO,deepPicker,topaz}
  cp -s ${SUB_DIR}/${LABEL}/cryolo/BOX/train/*.box ${TMP}/crYOLO/
  cp -s ${SUB_DIR}/${LABEL}/deep/BOX/train/*.box ${TMP}/deepPicker/
  cp -s ${SUB_DIR}/${LABEL}/topaz/BOX/train/*.box ${TMP}/topaz/
  repic get_cliques ${TMP} ${REPIC_OUT_DIR}/train ${REPIC_BOX_SIZE} &> ${REPIC_OUT_DIR}/clique_train.log
  repic run_ilp ${REPIC_OUT_DIR}/train ${REPIC_BOX_SIZE} --num_particles ${REPIC_NUM_PARTICLES} &> ${REPIC_OUT_DIR}/ilp_train.log
  rm -rf ${TMP}
  # val consensus
  mkdir -p ${TMP}/{crYOLO,deepPicker,topaz}
  cp -s ${SUB_DIR}/${LABEL}/cryolo/BOX/val/*.box ${TMP}/crYOLO/
  cp -s ${SUB_DIR}/${LABEL}/deep/BOX/val/*.box ${TMP}/deepPicker/
  cp -s ${SUB_DIR}/${LABEL}/topaz/BOX/val/*.box ${TMP}/topaz/
  repic get_cliques ${TMP} ${REPIC_OUT_DIR}/val ${REPIC_BOX_SIZE} &> ${REPIC_OUT_DIR}/clique_val.log
  repic run_ilp ${REPIC_OUT_DIR}/val ${REPIC_BOX_SIZE} --num_particles ${REPIC_NUM_PARTICLES} &> ${REPIC_OUT_DIR}/ilp_val.log
  rm -rf ${TMP}
  # test consensus
  mkdir -p ${TMP}/{crYOLO,deepPicker,topaz}
  cp -s ${SUB_DIR}/${LABEL}/cryolo/BOX/test/*.box ${TMP}/crYOLO/
  cp -s ${SUB_DIR}/${LABEL}/deep/BOX/test/*.box ${TMP}/deepPicker/
  cp -s ${SUB_DIR}/${LABEL}/topaz/BOX/test/*.box ${TMP}/topaz/
  repic get_cliques ${TMP} ${REPIC_OUT_DIR}/test ${REPIC_BOX_SIZE} &> ${REPIC_OUT_DIR}/clique_test.log
  repic run_ilp ${REPIC_OUT_DIR}/test ${REPIC_BOX_SIZE} --num_particles ${REPIC_NUM_PARTICLES} &> ${REPIC_OUT_DIR}/ilp_test.log
  rm -rf ${TMP}
  if ${GET_SCORE}; then
    python ${REPIC_UTILS}/score_detections.py -g ${REPIC_COORD}/train/${LABEL}/*.box -p ${REPIC_OUT_DIR}/train/*.box &> ${REPIC_OUT_DIR}/score_train.log
    # update Topaz pos-unlabeled mini-batch balancing with respect to training data
    export TOPAZ_BALANCE=$(awk '(NR > 1){total += $NF}END{print total/(NR - 1)}' ${REPIC_OUT_DIR}/train/particle_set_comp.tsv)
    python ${REPIC_UTILS}/score_detections.py -g ${REPIC_COORD}/val/*.box -p ${REPIC_OUT_DIR}/val/*.box &> ${REPIC_OUT_DIR}/score_val.log
    python ${REPIC_UTILS}/score_detections.py -g ${REPIC_COORD}/test/*.box -p ${REPIC_OUT_DIR}/test/*.box &> ${REPIC_OUT_DIR}/score_test.log
  fi
else
  rm -rf ${SUB_DIR}/${LABEL}/manual/{train,val}
  mkdir -p ${SUB_DIR}/${LABEL}/manual/{train,val}
  #	downsample and round (for Topaz) training labels
  REPIC_OUT_DIR=${SUB_DIR}/${LABEL}/manual/train
  touch ${REPIC_OUT_DIR}/tmp.box
  while [ $(cat ${REPIC_OUT_DIR}/*.box | wc -l) -le 4 ]; do	#	ensure there are enough examples sampled
    for file in ${REPIC_TRAIN_MRC}/*.box; do
      base=$(basename ${file})
      cat ${file} | awk 'BEGIN {srand()} !/^$/ { if (rand() <= 0.01) print $0}' > ${REPIC_OUT_DIR}/${base}
      perl -i -pe 's/(\d*\.\d*)/int($1+0.5)/ge' ${REPIC_OUT_DIR}/${base}
    done
  done
  #	remove empty files (leads to Topaz error)
  find ${REPIC_OUT_DIR}/ -type f -empty -delete
  #	downsample and round (for Topaz) validation labels
  REPIC_OUT_DIR=${SUB_DIR}/${LABEL}/manual/val
  touch ${REPIC_OUT_DIR}/tmp.box
  while [ $(cat ${REPIC_OUT_DIR}/*.box | wc -l) -le 4 ]; do
    for file in ${REPIC_VAL_MRC}/*.box; do
      base=$(basename ${file})
      cat ${file} | awk 'BEGIN {srand()} !/^$/ { if (rand() <= 0.01) print $0}' > ${REPIC_OUT_DIR}/${base}
      perl -i -pe 's/(\d*\.\d*)/int($1+0.5)/ge' ${REPIC_OUT_DIR}/${base}
    done
  done
  find ${REPIC_OUT_DIR}/ -type f -empty -delete
  export DEEP_BATCH_SIZE=4
fi

# #
#  step 3 - iteratively retrain algorithms using consenus particles as training labels
# #

i=0
while [ ${i} -lt ${NUM_ROUNDS} ]; do

  #  set input coordinate and output directories
  COORD_DIR=${IN_DIR}/iterative_particle_picking/round_${i}
  if ${USE_MANUAL_LABELS} && [ ${i} -eq 0 ]; then
    echo -e "\nFitting models to downsampled manual train and val labels ... "
    export REPIC_TRAIN_COORD=${COORD_DIR}/${LABEL}/manual/train/
    export REPIC_VAL_COORD=${COORD_DIR}/${LABEL}/manual/val/
  else
    echo -e "\nFitting models to estimated train and val labels ... "
    export REPIC_TRAIN_COORD=${COORD_DIR}/${LABEL}/clique_files/train/
    export REPIC_VAL_COORD=${COORD_DIR}/${LABEL}/clique_files/val/
  fi
  #	uncomment line below if continuing from iteration >1
  # export TOPAZ_BALANCE=$(awk '(NR > 1){total += $NF}END{print total/(NR - 1)}' ${COORD_DIR}/${LABEL}/clique_files/train/particle_set_comp.tsv)
  i=$(( $i + 1 ))
  SUB_DIR=${IN_DIR}/iterative_particle_picking/round_${i}
  echo "--- Round: ${i} ---"
  echo -e "--- Identifying particles in '${LABEL}' ---\n\tSPHIRE-crYOLO ... "
  rm -rf ${REPIC_TRAIN_COORD}/STAR ${REPIC_VAL_COORD}/STAR	#	precaution - to prevent CrYOLO from learning labels with 2x weight

  #  SPHIRE-crYOLO
  # fit model
  export REPIC_OUT_DIR=${SUB_DIR}/${LABEL}/cryolo
  mkdir -p ${REPIC_OUT_DIR}/BOX/{train,val,test}
  bash ${REPIC}/iterative_particle_picking/fit_cryolo.sh &> ${REPIC_OUT_DIR}/iter_fit.log
  # train set prediction
  export REPIC_MRC_DIR=${REPIC_TRAIN_MRC}
  export CRYOLO_MODEL=${REPIC_OUT_DIR}/learned_weights.h5
  rm -rf ${REPIC_OUT_DIR}/{CBOX,STAR}/*
  bash ${REPIC}/iterative_particle_picking/run_cryolo.sh &> ${REPIC_OUT_DIR}/iter_train.log
  python ${REPIC_UTILS}/coord_converter.py ${REPIC_OUT_DIR}/CBOX/*.cbox ${REPIC_OUT_DIR}/BOX/train/ -f cbox -t box -b ${REPIC_BOX_SIZE} --round 0 --force &> ${REPIC_OUT_DIR}/convert_train.log
  #  val set prediction
  export REPIC_MRC_DIR=${REPIC_VAL_MRC}
  rm -rf ${REPIC_OUT_DIR}/{CBOX,STAR}/*
  bash ${REPIC}/iterative_particle_picking/run_cryolo.sh &> ${REPIC_OUT_DIR}/iter_val.log
  python ${REPIC_UTILS}/coord_converter.py ${REPIC_OUT_DIR}/CBOX/*.cbox ${REPIC_OUT_DIR}/BOX/val/ -f cbox -t box -b ${REPIC_BOX_SIZE} --round 0 --force &> ${REPIC_OUT_DIR}/convert_val.log
  #  test set prediction
  export REPIC_MRC_DIR=${REPIC_TEST_MRC}
  rm -rf ${REPIC_OUT_DIR}/{CBOX,STAR}/*
  bash ${REPIC}/iterative_particle_picking/run_cryolo.sh &> ${REPIC_OUT_DIR}/iter_test.log
  python ${REPIC_UTILS}/coord_converter.py ${REPIC_OUT_DIR}/CBOX/*.cbox ${REPIC_OUT_DIR}/BOX/test/ -f cbox -t box -b ${REPIC_BOX_SIZE} --round 0 --force &> ${REPIC_OUT_DIR}/convert_test.log
  if ${GET_SCORE}; then
    python ${REPIC_UTILS}/score_detections.py -g ${REPIC_COORD}/train/${LABEL}/*.box -p ${REPIC_OUT_DIR}/BOX/train/*.box -c 0.3 &> ${REPIC_OUT_DIR}/score_train.log
    python ${REPIC_UTILS}/score_detections.py -g ${REPIC_COORD}/val/*.box -p ${REPIC_OUT_DIR}/BOX/val/*.box -c 0.3 &> ${REPIC_OUT_DIR}/score_val.log
    python ${REPIC_UTILS}/score_detections.py -g ${REPIC_COORD}/test/*.box -p ${REPIC_OUT_DIR}/BOX/test/*.box -c 0.3 &> ${REPIC_OUT_DIR}/score_test.log
  fi

  echo -e "\tDeepPicker ... "
  #   DeepPicker
  #   fit model
  export REPIC_OUT_DIR=${SUB_DIR}/${LABEL}/deep
  mkdir -p ${REPIC_OUT_DIR}/BOX/{train,val,test}
  bash ${REPIC}/iterative_particle_picking/fit_deep.sh &> ${REPIC_OUT_DIR}/iter_fit.log
  # #   train set prediction
  export REPIC_MRC_DIR=${REPIC_TRAIN_MRC}
  export DEEP_MODEL=${REPIC_OUT_DIR}/model_demo_type3_refined
  rm -rf ${REPIC_OUT_DIR}/STAR/*.star
  bash ${REPIC}/iterative_particle_picking/run_deep.sh &> ${REPIC_OUT_DIR}/iter_train.log
  python ${REPIC_UTILS}/coord_converter.py ${REPIC_OUT_DIR}/STAR/*.star ${REPIC_OUT_DIR}/BOX/train/ -f star -t box -b ${REPIC_BOX_SIZE} --round 0 --force &> ${REPIC_OUT_DIR}/convert_train.log
  #  val set prediction
  export REPIC_MRC_DIR=${REPIC_VAL_MRC}
  rm -rf ${REPIC_OUT_DIR}/STAR/*.star
  bash ${REPIC}/iterative_particle_picking/run_deep.sh &> ${REPIC_OUT_DIR}/iter_val.log
  python ${REPIC_UTILS}/coord_converter.py ${REPIC_OUT_DIR}/STAR/*.star ${REPIC_OUT_DIR}/BOX/val/ -f star -t box -b ${REPIC_BOX_SIZE} --round 0 --force &> ${REPIC_OUT_DIR}/convert_val.log
  #  test set prediction
  export REPIC_MRC_DIR=${REPIC_TEST_MRC}
  rm -rf ${REPIC_OUT_DIR}/STAR/*.star
  bash ${REPIC}/iterative_particle_picking/run_deep.sh &> ${REPIC_OUT_DIR}/iter_test.log
  python ${REPIC_UTILS}/coord_converter.py ${REPIC_OUT_DIR}/STAR/*.star ${REPIC_OUT_DIR}/BOX/test/ -f star -t box -b ${REPIC_BOX_SIZE} --round 0 --force &> ${REPIC_OUT_DIR}/convert_test.log
  if ${GET_SCORE}; then
    python ${REPIC_UTILS}/score_detections.py -g ${REPIC_COORD}/train/${LABEL}/*.box -p ${REPIC_OUT_DIR}/BOX/train/*.box -c 0.5 &> ${REPIC_OUT_DIR}/score_train.log
    python ${REPIC_UTILS}/score_detections.py -g ${REPIC_COORD}/val/*.box -p ${REPIC_OUT_DIR}/BOX/val/*.box -c 0.5 &> ${REPIC_OUT_DIR}/score_val.log
    python ${REPIC_UTILS}/score_detections.py -g ${REPIC_COORD}/test/*.box -p ${REPIC_OUT_DIR}/BOX/test/*.box -c 0.5 &> ${REPIC_OUT_DIR}/score_test.log
  fi

  echo -e "\tTopaz ... "
  #  Topaz
  #  fit model
  export REPIC_OUT_DIR=${SUB_DIR}/${LABEL}/topaz
  mkdir -p ${REPIC_OUT_DIR}/BOX/{train,val,test}
  rm -f ${REPIC_OUT_DIR}/predicted_particles_all_upsampled.txt	#	incase of prev. instance (shouldn't be needed)
  bash ${REPIC}/iterative_particle_picking/fit_topaz.sh &> ${REPIC_OUT_DIR}/iter_fit.log
  # train set prediction
  export REPIC_MRC_DIR=${REPIC_TRAIN_MRC}
  export TOPAZ_MODEL=${REPIC_OUT_DIR}/model_epoch10.sav
  rm -f ${REPIC_OUT_DIR}/predicted_particles_all_upsampled.txt
  bash ${REPIC}/iterative_particle_picking/run_topaz.sh &> ${REPIC_OUT_DIR}/iter_train.log
  python ${REPIC_UTILS}/coord_converter.py ${REPIC_OUT_DIR}/predicted_particles_all_upsampled.txt ${REPIC_OUT_DIR}/BOX/train -f tsv -t box -b ${REPIC_BOX_SIZE} -c 1 2 none none 3 0 --header --multi_out --round 0 --force &> ${REPIC_OUT_DIR}/convert_train.log
  # val set prediction
  export REPIC_MRC_DIR=${REPIC_VAL_MRC}
  rm -f  ${REPIC_OUT_DIR}/predicted_particles_all_upsampled.txt
  bash ${REPIC}/iterative_particle_picking/run_topaz.sh &> ${REPIC_OUT_DIR}/iter_val.log
  python ${REPIC_UTILS}/coord_converter.py ${REPIC_OUT_DIR}/predicted_particles_all_upsampled.txt ${REPIC_OUT_DIR}/BOX/val -f tsv -t box -b ${REPIC_BOX_SIZE} -c 1 2 none none 3 0 --header --multi_out --round 0 --force &> ${REPIC_OUT_DIR}/convert_val.log
  # test set prediction
  export REPIC_MRC_DIR=${REPIC_TEST_MRC}
  rm -f ${REPIC_OUT_DIR}/predicted_particles_all_upsampled.txt
  bash ${REPIC}/iterative_particle_picking/run_topaz.sh &> ${REPIC_OUT_DIR}/iter_test.log
  python ${REPIC_UTILS}/coord_converter.py ${REPIC_OUT_DIR}/predicted_particles_all_upsampled.txt ${REPIC_OUT_DIR}/BOX/test -f tsv -t box -b ${REPIC_BOX_SIZE} -c 1 2 none none 3 0 --header --multi_out --round 0 --force &> ${REPIC_OUT_DIR}/convert_test.log
  if ${GET_SCORE}; then
    python ${REPIC_UTILS}/score_detections.py -g ${REPIC_COORD}/train/${LABEL}/*.box -p ${REPIC_OUT_DIR}/BOX/train/*.box -c 0. &> ${REPIC_OUT_DIR}/score_train.log
    python ${REPIC_UTILS}/score_detections.py -g ${REPIC_COORD}/val/*.box -p ${REPIC_OUT_DIR}/BOX/val/*.box -c 0. &> ${REPIC_OUT_DIR}/score_val.log
    python ${REPIC_UTILS}/score_detections.py -g ${REPIC_COORD}/test/*.box -p ${REPIC_OUT_DIR}/BOX/test/*.box -c 0. &> ${REPIC_OUT_DIR}/score_test.log
  fi

  echo -e "\tBuilding consensus ... "
  TMP=${SUB_DIR}/${LABEL}/tmp
  rm -rf ${TMP}
  # train consensus
  mkdir -p ${TMP}/{crYOLO,deepPicker,topaz}
  cp -fs ${SUB_DIR}/${LABEL}/cryolo/BOX/train/*.box ${TMP}/crYOLO/
  cp -fs ${SUB_DIR}/${LABEL}/deep/BOX/train/*.box ${TMP}/deepPicker/
  cp -fs ${SUB_DIR}/${LABEL}/topaz/BOX/train/*.box ${TMP}/topaz/
  REPIC_OUT_DIR=${SUB_DIR}/${LABEL}/clique_files
  mkdir -p ${REPIC_OUT_DIR}
  repic get_cliques ${TMP} ${REPIC_OUT_DIR}/train ${REPIC_BOX_SIZE} &> ${REPIC_OUT_DIR}/clique_train.log
  repic run_ilp ${REPIC_OUT_DIR}/train ${REPIC_BOX_SIZE} --num_particles ${REPIC_NUM_PARTICLES} &> ${REPIC_OUT_DIR}/ilp_train.log
  rm -rf ${TMP}
  # val consensus
  mkdir -p ${TMP}/{crYOLO,deepPicker,topaz}
  cp -s ${SUB_DIR}/${LABEL}/cryolo/BOX/val/*.box ${TMP}/crYOLO/
  cp -s ${SUB_DIR}/${LABEL}/deep/BOX/val/*.box ${TMP}/deepPicker/
  cp -s ${SUB_DIR}/${LABEL}/topaz/BOX/val/*.box ${TMP}/topaz/
  repic get_cliques ${TMP} ${REPIC_OUT_DIR}/val ${REPIC_BOX_SIZE} &> ${REPIC_OUT_DIR}/clique_val.log
  repic run_ilp ${REPIC_OUT_DIR}/val ${REPIC_BOX_SIZE} --num_particles ${REPIC_NUM_PARTICLES} &> ${REPIC_OUT_DIR}/ilp_val.log
  rm -rf ${TMP}
  # test consensus
  mkdir -p ${TMP}/{crYOLO,deepPicker,topaz}
  cp -s ${SUB_DIR}/${LABEL}/cryolo/BOX/test/*.box ${TMP}/crYOLO/
  cp -s ${SUB_DIR}/${LABEL}/deep/BOX/test/*.box ${TMP}/deepPicker/
  cp -s ${SUB_DIR}/${LABEL}/topaz/BOX/test/*.box ${TMP}/topaz/
  repic get_cliques ${TMP} ${REPIC_OUT_DIR}/test ${REPIC_BOX_SIZE} &> ${REPIC_OUT_DIR}/clique_test.log
  repic run_ilp ${REPIC_OUT_DIR}/test ${REPIC_BOX_SIZE} --num_particles ${REPIC_NUM_PARTICLES} &> ${REPIC_OUT_DIR}/ilp_test.log
  rm -rf ${TMP}
  if ${GET_SCORE}; then
    python ${REPIC_UTILS}/score_detections.py -g ${REPIC_COORD}/train/${LABEL}/*.box -p ${REPIC_OUT_DIR}/train/*.box &> ${REPIC_OUT_DIR}/score_train.log
    export TOPAZ_BALANCE=$(awk '(NR > 1){total += $NF}END{print total/(NR - 1)}' ${REPIC_OUT_DIR}/train/particle_set_comp.tsv)
    python ${REPIC_UTILS}/score_detections.py -g ${REPIC_COORD}/val/*.box -p ${REPIC_OUT_DIR}/val/*.box &> ${REPIC_OUT_DIR}/score_val.log
    python ${REPIC_UTILS}/score_detections.py -g ${REPIC_COORD}/test/*.box -p ${REPIC_OUT_DIR}/test/*.box &> ${REPIC_OUT_DIR}/score_test.log
  fi
  export DEEP_BATCH_SIZE=32	#	in case manual particles are used with downsampling

done
