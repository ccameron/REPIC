"""
--> This module provides the functionality to predict the positions of transmembrane - proteins in one whole tomogram.
    The Tomogram is expected to be a mrc-file and the result is a .pos-file with the same name.
    Each line of the pos-file contains the coordinates of one predicted protein. In the following order (x, y, z).
    (same order as given in the mrc-file)
    Each 3D-Coordinate represents a midpoint of a prediction.
--> Furthermore, this module contains functionality for evaluating the predicted result with a given pos-file.
    Recall, Precision, F1 and some other information to describe the difference between the prediction and the
    ground truth is determined.

Authors: Erik Genthe, Philipp Heuser.
Refactor: Indira Tekkali
"""
import os as OS
import sys as SYS
import argparse as ARG
import numpy as NP
import torch as T
import sklearn.cluster
import modules.tools as TOOLS
import modules.pickyoloNet as NET
from time import time


class Prediction:
    """ The main class of this module. See module-documentation for more information. """

    def __init__(self, eval_threshold_distance, voxel_size):
        """
        :param eval_threshold_distance: Maximum distance between a ground-truth object and a predicted object that
                                        implies a correct prediction (True positive).
        :param voxel_size: A tuple with three values, which defines th relative size of a voxel on each axis.
        """
        self.use_gpu = True
        self.tomo_depth = None
        self.tomo_width = None
        self.tomo_height = None
        self.eval_threshold_distance = eval_threshold_distance
        self.voxel_size = voxel_size
        self.obj_thres = 0.5    #   REPIC_PATCH

    def setArgs(self, input_path: str, model_path: str, result_path: str, ground_truth_file_path: str = None,
                vis_pred= True):
        """
        Validate arguments and paste them to self
        param input_path: Input file path, either can be a single .mrc file or a folder contain multiple files.
        :param model_path: Trained model path
        :param result_path: To save the resulting file in the specified path
        :param ground_truth_file_path: The path of the ground-truth-file. Is only required, if you want to do automatic
                                       evaluation in the end.
        :return: None
        """
        self.input_path = input_path
        assert OS.path.isdir(input_path) or input_path.endswith('.mrc')
        if ground_truth_file_path is None:
            self.eval_requested = False
        else:
            if not OS.path.isdir(ground_truth_file_path) and not ground_truth_file_path.lower().endswith('.pos'):
                print("Warning: The input-file is expected to have the suffix .pos." + "which it currently has not")
            self.ground_truth_file_path = ground_truth_file_path
            self.eval_requested = True

        assert OS.path.isdir(result_path), "The specified result-path has to be a directory."
        self.result_dir = result_path
        self.result_path = result_path
        inputFileName = self.input_path.rsplit("/", 1)[1]
        outputFileName = inputFileName.rsplit(".", 1)[0] + ".pos"
        self.result_path += outputFileName
        self.model_path = model_path
        self.vis_pred = vis_pred


    def parseArgs(self):
        description = """\nMakes predictions on a mrc-file resultant predictions are saved to a .pos-file.
        \nwhich contains a list of 3D-coordinates, represents the midpoints of the predicted proteins.
        \nThe name of the file will be the same as the given mrc-file.
        \nIn case if --eval is specified. The result is compared with the given .pos file.
        """
        parser = ARG.ArgumentParser(prog='pickyolo.py configpath predict', description=description, formatter_class=
                                    ARG.RawTextHelpFormatter)
        parser.add_argument('input_path', type=str, help="The path of the mrc-file on which you want to apply predictions"
                            + "\nThe path can also be a directory. Then all .mrc-files in that directory will be used.")
        parser.add_argument('--result-path', type=str, default="./",help="The directory (without filename) where the "
                                            "compressed, prepared " + "dataset saved")
        parser.add_argument('--eval', type=str, help="If given, there will be an evaluation of the predicted result."
                        "\nAlso enter the path to a pos-file which can be used for comparison")
        parser.add_argument('--model', type=str, default="./model.pth", help="Specify the path to the trained model for"
                                                                             " inference.")
        parser.add_argument('--eval_images', action="store_true", default=False, help="Creates images for TP, Fp & FN")
        parser.add_argument('--obj_thres', type=float, default=0.5, help="Specify the OBJECTNESS_THRESHOLD for identifying 2D particle slices (default=0.5)")   #   REPIC_PATCH
        args = parser.parse_args(SYS.argv[3:])
        self.obj_thres = args.obj_thres    #   REPIC_PATCH
        del args.obj_thres                 #   REPIC_PATCH
        self.args = args
        self.setArgs(args.input_path, args.model, args.result_path, args.eval)

    def load(self, net=None):
        """
        Load the data that shall be used for evaluation, converts the data into a numpy array and normalize the data by
        mean and standard variance
        :param net: torch model
        :return: None
        """
        # Load input-data
        self.x_numpy = TOOLS.mrc_2_np(self.input_path)
        self.x_numpy = self.x_numpy.astype(NP.float32)

        shape = self.x_numpy.shape
        assert len(shape) == 3

        self.tomo_depth = shape[0]
        self.tomo_width = shape[1]
        self.tomo_height = shape[2]

        # Normalize
        self.x_numpy.flags.writeable = True
        self.x_numpy -= self.x_numpy.mean()
        self.x_numpy *= 1 / self.x_numpy.std()

        # To GPU?
        if self.use_gpu:
            self.x = T.cuda.FloatTensor(self.x_numpy)
        else:
            self.x = T.FloatTensor(self.x_numpy)
        self.x = self.x.reshape(shape[0], 1, shape[-2], shape[-1])      # Efficient. Data is not copied.
        # Init net
        if net is None:
            self.net = T.load(self.model_path)
        else:
            self.net = net
        if self.use_gpu:
            self.net = self.net.cuda()
        self.net.eval()  # set pytorch-evaluation-mode

        # Maintain backward-compatibility for older network versions.
        if not hasattr(self.net, 'dropout_rate'):
            self.net.dropout_rate = 0.0
            self.net.use_batchnorm = True
            self.net.backbone.dropout_rate = 0.0
            self.net.backbone.use_batchnorm = True


    def _cleanGPU(self, cleanupNet=True):
        """ Releasing the GPU-Memory """
        del self.x
        if cleanupNet:
            del self.net

    def predict2D(self, cleanupNet=True):
        """
        After loading the data, this method can be called to execute on the actual prediction.
        After this you can eventually execute eval() to evaluate the results.
        param cleanupNet: To release the GPU memory
        :return: None
        """
        assert self.x is not None, "Predict: load() has to be called before predict() can be called."
        x = self.x
        assert len(x.shape) == 4 and x.shape[1] == 1, "The shape is expected to be. (sliceAmount, 1, W, H)"
        z = None
        for s in range(x.shape[0]):  # here s is a slice
            zTensor = self.net(x[s:s+1]).cpu().detach()
            if z is None:
                z = NP.empty((x.shape[0], 3, zTensor.shape[-2], zTensor.shape[-1]))
            z[s:s+1] = NP.array(zTensor)
        x = NP.array(x.cpu())  # .astype(NP.float16)
        self.netOutput = z
        self._cleanGPU(cleanupNet)

    def clusterTo3D(self, min_predictions, eps, makeEvalImages=False, saveResultToFile=True):
        """
        Takes the raw-output of the neural-network (from self.netOutput) and returns a list of clustered 3D-Coordinates.
        :param min_predictions: see https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
        :param eps: See: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
        :param makeEvalImages: To save TP, FP and FN
        :param saveResultToFile: To save the model predictions into a .pos file
        :return: f1-score, TP, FP and FN
        """

        assert hasattr(self, 'netOutput')
        assert len(self.netOutput.shape) == 4 and self.netOutput.shape[1] == 3, "The shape is expected to be. \
                (sliceAmount, 3, outputW, outputH)"
        if self.tomo_depth is None:
            print("Warning. Could not assert correct image-size. Use predict.load() first")
        else:
            assert self.tomo_depth == self.netOutput.shape[0]

        # Contains the 2D-Predictions with 3D-Coordinates.
        unclusteredPredictions = []  # [(z, y, x)]
        clusteredPredictions = []

        for s in range(self.netOutput.shape[0]):
            preds = NET.extractPredictions(self.netOutput[s], inputSize=(self.tomo_width, self.tomo_height), objectness_threshold=self.obj_thres)  #   REPIC_PATCH
            for (x, y, conf) in preds:                                          #   REPIC_PATCH
                unclusteredPredictions.append((s, x, y, conf))                  #   REPIC_PATCH
        if unclusteredPredictions:
            unclusteredPredictions = NP.array(unclusteredPredictions)
            _, labels = sklearn.cluster.dbscan(unclusteredPredictions[:, :-1],  #   REPIC_PATCH
                                               eps,
                                               min_samples=min_predictions,
                                               algorithm='auto')
            clusters = NP.empty(labels.max()+1, dtype=list)
            for i in range(len(labels)):
                l = labels[i]
                if l == -1:  # Ignore noise
                    continue
                if clusters[l] is None:
                    clusters[l] = []
                clusters[l].append(unclusteredPredictions[i])

            for cluster in clusters:
                cluster = NP.array(cluster)
                cluster = cluster.swapaxes(0, 1)
                mid_x = cluster[0].mean()
                mid_y = cluster[1].mean()
                mid_z = cluster[2].mean()
                mid_conf = cluster[3].mean()                                    #   REPIC_PATCH
                clusteredPredictions.append((mid_x, mid_y, mid_z, mid_conf))    #   REPIC_PATCH

            clusteredPredictions.sort(key = lambda x: x[:-1])                    #   REPIC_PATCH

        if saveResultToFile:
            TOOLS.list_2_pos(self.result_path, clusteredPredictions)

        if self.eval_requested:
            """ Executes evaluation
                :return: f1-score, amount of true positives, false positives, false negatives
            """
            assert isinstance(self.ground_truth_file_path, str)

            # Cache ground_truth from file.
            if not hasattr(self, 'ground_truth'):
                self.ground_truth = TOOLS.pos_2_np(self.ground_truth_file_path, only_midpoints=True)
            if makeEvalImages:
                namePrefix = self.input_path.rsplit('.', 1)[0].rsplit('/', 1)[1]
                tp, fp, fn = TOOLS.visualizeTpFpFn(self.x_numpy, self.ground_truth, clusteredPredictions,
                                                   self.eval_threshold_distance, self.voxel_size, namePrefix)
            else:
                tp, fp, fn = TOOLS.getConfusionMetrics(self.ground_truth, clusteredPredictions,
                                                       self.eval_threshold_distance, self.voxel_size)
            f1 = 0
            if tp > 0:
               f1 = tp / (tp + 0.5 * (fp+fn))
            return f1, tp, fp, fn


def main(config:dict):
    """ Executes a whole prediction """

    def doPredictionWorkflow(predictionInstance: Prediction):
        predictionInstance.load()  # data loading
        t_start = time()

        # Store these values before self.net gets cleaned from gpu
        min_pred = predictionInstance.net.this_min_pred
        eps = predictionInstance.net.this_eps
        print(f"Loaded cluster params: min_pred = {min_pred}, eps = {eps}")

        # In the end of this self.net gets cleaned from gpu
        predictionInstance.predict2D()  # 2d particle detection

        evalResult = predictionInstance.clusterTo3D(min_pred, eps,
                                                    makeEvalImages=predictionInstance.args.eval_images)
        print("Prediction + optional evaluation took ", time() - t_start, "s")
        if evalResult:
            print("Evaluation result: f1=%3f   tp=%d   fp=%d   fn=%d" % evalResult)

    assert isinstance(config, dict)

    # Check that all required keys are set in config. Give useful error-msg if not.
    requiredKeysFromConfig = ['EVAL_THRESHOLD_DISTANCE', 'VOXEL_SIZE']
    for key in requiredKeysFromConfig:
        assert key in config, key + " has to be defined in config-file."

    if 'CLUSTER_MIN_PREDICTIONS' in config or 'CLUSTER_EPS' in config:
        print("Warning: Config entries 'CLUSTER_MIN_PREDICTIONS' and 'CLUSTER_EPS' are ignored.",
            "Those values are part of the saved model now and loaded from there.")

    p = Prediction(config['EVAL_THRESHOLD_DISTANCE'], config['VOXEL_SIZE'])
    p.parseArgs()
    if OS.path.isdir(p.input_path):  # If inputpath is a directory
        inputDir = p.input_path
        mrcfiles = list(filter(lambda p: ".mrc" in p.lower(), OS.listdir(inputDir)))
        if p.eval_requested:
            if not OS.path.isdir(p.args.eval):
                raise RuntimeError("If a directory is specified as data-path, the eval-parameter has to be a directory "
                                   "as well." + "The directory, that contains the annotation-files.")
        for currentfile in mrcfiles:
            p.input_path = OS.path.join(inputDir, currentfile)
            p.result_path = OS.path.join(p.result_dir, currentfile.replace('.mrc', '.pos'))
            if p.eval_requested:
                if hasattr(p, 'ground_truth'):
                    del p.ground_truth
                p.ground_truth_file_path = OS.path.join(p.args.eval, currentfile.replace('.mrc', '.pos'))
            print("Predicting on: ", currentfile)
            doPredictionWorkflow(p)

    else:   # If inputpath was a single file
        doPredictionWorkflow(p)
