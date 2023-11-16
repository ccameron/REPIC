"""
This module contains various helper functions.
Authors: Erik Genthe, Philipp Heuser.
Refactor: Indira Tekkali
"""
import os
import mrcfile as MRC
import numpy as np
import threading as LOCK
import PIL
from PIL import ImageDraw, Image, ImageOps
import numba as NUMBA

fileSystemLock = LOCK.Lock()


def mkdir(path: str):
    fileSystemLock.acquire()
    if not os.path.exists(path):
        os.makedirs(path)
    fileSystemLock.release()


def voxel_dist(p1, p2, voxelSize):
    dx = voxelSize[0] * (p1[2] - p2[2])
    dy = voxelSize[1] * (p1[1] - p2[1])
    dz = voxelSize[2] * (p1[0] - p2[0])
    return (dx**2 + dy**2 + dz**2)**0.5


def getConfusionMetrics(ground_truth: list, predictions: list, eval_threshold_distance: float, voxelSize):
    """
    Counts the amount of true positives, false positives, false negatives.
    If there are two coordinates for an entry, the midpoint is calculated and used.
    param ground_truth: The path of the file, which contains the ground-truth coordinates.
    param predictions: Network raw predictions.
    param eval_threshold_distance: Maximum distance between a ground-truth object and a predicted object that implies
                                    a correct prediction (True positive).
    param voxelSize: A tuple with three values, which defines th relative size of a voxel on each axis.
    return: rue positives, false positives, false negatives.
    """
    assert isinstance(eval_threshold_distance, (float, int))
    assert voxelSize is not None and len(voxelSize) == 3

    ground_truth = list(ground_truth)
    tp, fp, fn = 0, 0, 0  # true positives, false positives, false negatives
    for pred in predictions:
        isTP = False
        for i in range(len(ground_truth)):
            if voxel_dist(pred, ground_truth[i], voxelSize) <= eval_threshold_distance:
                tp += 1
                isTP = True
                ground_truth.pop(i)
                break
        if not isTP:
            fp += 1
    fn = len(ground_truth)
    return tp, fp, fn


def tomoToUint8(tomogram, contrast_multiplier=0.25):
    """
    Takes a tomogram or any other numpy-array as input, normalizes it, scales and shifts the values to fit 0-255-Space
    and converts them to numpy.uint8-Values.
    param tomogram: The input-data (multidimensional float-numpy-array is expected).
    param contrast_multiplier: Scales the contrast.
    :return: A new numpy-array, with the same dimensions, but numpy.uint8 as type.
    """
    # Normalize
    tomogram -= tomogram.mean()
    tomogram /= tomogram.std()

    # Convert to uint8
    tomogram *= contrast_multiplier * 256 / 2
    tomogram += 256 / 2
    tomogram = tomogram.clip(0, 255)
    tomogram = tomogram.astype(np.uint8)
    return tomogram


def visualizeSingleAnnotation(tomogram: np.array, annotation: tuple, path: str, convertData=True, cutout_width=128,
                              cross_size=5, z_range=(-10, -5, 0, 5, 10)):
    """
    Creates a 2d-image containing multiple slices of a particle, which is cropped from thegiven tomogram. The image
    gets immediately saved to a file.
    param tomogram: Numpy-array-like input-data. 3 dimensions expected.
    param annotation: A triple (z, y, x), that contains the pixel-space-coordinates of the midpoint, of the region,
                      that shall be cropped from the tomogram.
    param path: The path for the new image. The image-type can be changed by the suffix.
    param convertData: If True, tomogram will be normalized, color intensities will be shifted and scaled.
                        See tomoToUint8(...)
    param cutout_width: The width and height of the area to crop out in pixels.
    param cross_size: In the middel of each slice a small cross is drawn. This is the width or height of this cross.
    param z_range: For each element of this tuple a slice is extracted from the tomogram. The element specifies the
                    z-index relative to the annotation.
    :return: None
    """
    hw = cutout_width / 2
    x = int(annotation[2])
    y = int(annotation[1])
    z = int(annotation[0])
    img = PIL.Image.new(mode='L', size=(len(z_range) * (cutout_width+1) - 1, cutout_width))
    if convertData:
        tomogram = tomoToUint8(tomogram)
    for j in range(0, len(z_range)):
        z_local = z + z_range[j]
        # Create image for a single slice
        if z_local < 0 or z_local >= tomogram.shape[0]:
            continue
        img_part = PIL.Image.fromarray(tomogram[z_local], mode='L')
        img_part = img_part.transform(size=(cutout_width, cutout_width), method=PIL.Image.EXTENT,
                                      data=(x-hw, y-hw, x+hw, y+hw), resample=0, fill=1, fillcolor=None)
        # Draw box in the middle of the slice
        draw = ImageDraw.Draw(img_part)
        color = 255
        draw.line((hw-cross_size, hw, hw+cross_size, hw), fill=color, width=1)
        draw.line((hw, hw-cross_size, hw, hw+cross_size), fill=color, width=1)
        # Paste the slice to its position in the full image
        img.paste(img_part, box=(j * (cutout_width+1),  0,  (j+1) * (cutout_width+1) - 1,  cutout_width))
    img.save(path)


def visualizeTpFpFn(tomogram: np.array, ground_truth: list, predictions: list, eval_threshold_distance: float,
                    voxelSize: tuple, filenamePrefix = ''):
    """
    Creates cropped images, with mulitples slices for all true positives, false positives, and false negatives, using
    visualizeSingleAnnotation(...). These images a stored in different pathes: ./tp/, ./fp/, ./fn/. The folders are
    automatically created.
    param tomogram: A 3d-numpy-array, that contains the tomogram.
    param ground_truth: A list that contains the ground-truth coordinates.
    param predictions: A file path, which contains the predicted coordinates.
    param eval_threshold_distance: Maximum distance between a ground-truth object and a predicted object that implies a
                                    correct prediction (True positive).
    param voxelSize:  A tuple with three values, which defines th relative size of a voxel on each axis.
    param filenamePrefix: A string that will be appended to the image filenames.
    :return: true positives, false positives, false negatives
    """
    assert isinstance(eval_threshold_distance, (float, int))
    assert voxelSize is not None and len(voxelSize) == 3

    pathTP = "tp/"
    pathFP = "fp/"
    pathFN = "fn/"

    mkdir(pathTP)
    mkdir(pathFP)
    mkdir(pathFN)

    predictions = list(predictions)
    ground_truth = list(ground_truth)
    tomogram = tomoToUint8(tomogram)

    # Calculate true positives, fp, fn...
    tp, fp, fn = 0, 0, 0  # true positives, false positives, false negatives

    def saveImage(folder, ann):
        coordstr = "x%d_y%d_z%d" % (ann[2], ann[1], ann[0])
        path = folder + filenamePrefix + '__' + coordstr + '.png'
        visualizeSingleAnnotation(tomogram, ann, path, convertData=False)

    for pred in predictions:
        isTP = False
        for i in range(len(ground_truth)):
            if voxel_dist(pred, ground_truth[i], voxelSize) <= eval_threshold_distance:
                saveImage(pathTP, pred)
                tp += 1
                isTP = True
                ground_truth.pop(i)
                break
        if not isTP:
            saveImage(pathFP, pred)
            fp += 1
    fn = len(ground_truth)
    for annotation in ground_truth:
        saveImage(pathFN, annotation)

    return tp, fp, fn


def createMrcWithAnnotations(tomogram: np.array, annotations: list, gt_anno:list, file_path: str):
    """
    Creates a mrc-file. Where the annotation is painted in as white crosses.
    param tomogram: A numpy-array, which represents the whole data of the tomogram. Grayscale only.
                     Expected shape: (depth, width, height).
    param annotations: A list or numpy-array, that contains tuples or arrays representing 3D-Coordinates in pixel-Space
    param file_path: Filepath to store the result.
    :return: None
    """

    assert len(tomogram.shape) == 3, "Got tomogram with shape: " + str(tomogram.shape)
    assert isinstance(annotations, (list, np.ndarray))
    if len(annotations) > 0:
        assert len(annotations[0]) == 3
    assert isinstance(file_path, str)

    ANNOTATION_THICKNESS = 5  # On how many slices shall each annotation be painted to.
    COLOR = tomogram.max()
    cross_size = 10
    def draw_ann(ann, shift, color):
        for a in ann:
            for i in range(ANNOTATION_THICKNESS):
                slice_ = int(a[0] + i - ANNOTATION_THICKNESS // 2)
                slice_ = min(max(0, slice_), tomogram.shape[0]-1)
                im = PIL.Image.fromarray(tomogram[slice_])
                # im = PIL.Image.fromarray(tomogram[pic])
                draw = PIL.ImageDraw.Draw(im)
                # draw.line((a[2], a[1] - cross_size, a[2], a[1] + cross_size), fill= (255, 0, 0))
                # draw.line((a[2] - cross_size, a[1], a[2] + cross_size, a[1]), fill= (255, 0, 0))
                bbox_hw = int(0.5*32)
                coord = (a[2]-bbox_hw, a[1]-bbox_hw, a[2]+bbox_hw+shift, a[1]+bbox_hw+shift)
                draw.rectangle(coord, outline=color)
                tomogram[slice_] = np.asarray(im)
    draw_ann(annotations, shift=0, color=255)
    draw_ann(gt_anno, shift=1, color=0)
    mrc = MRC.new(file_path+'.mrc', overwrite=True)
    mrc.set_data(tomogram)
    mrc.close()


def showPosOnSlice(slice, targets, boxWidth: float, targets2=None, prefix=""):
    """
    Plotting the bounding boxes on the given slice
    param slice: A type of NP array or a torch tensor
    param targets: which shall be drawn as quads into the image (pixel-space).
    param boxWidth: Width of the bounding box
    param targets2: Similar to the targets but these target Box's are 1px bigger in width and height as you would
                     expect. The benefit is, that you can see both boxes, if there are exactly on the same position.
    param prefix: A prefix for the filename.
    :return: None
    """
    slice_ = np.array(slice[0])  # Gibt nur graustufen...
    imgWidth = len(slice_)
    imgHeight = len(slice_[0])
    slicePic = (20*slice_+(255//2))  # !!!! Here is the colorspace defined!
    slicePic = np.clip(slicePic, 0, 255)
    pic = np.empty((imgWidth, imgHeight, 3))
    pic[:, :, 0] = slicePic
    pic[:, :, 1] = slicePic
    pic[:, :, 2] = slicePic
    img = PIL.Image.fromarray(pic.astype(np.uint8))

    draw = ImageDraw.Draw(img)
    if targets2 is None:
        targets2 = []
    targets.sort()
    targets2.sort()
    # Print Big-Targets first.
    def draw_bbox(tar_list, bb_width=boxWidth, shift=0, color_normal=(0, 255, 0),
                  color_big=(0, 255, 255)):
        for element in tar_list:
            color = color_normal
            if len(element) > 2 and element[2] == 'big':
                color = color_big
            tar_ = (int(element[0]), int(element[1]))
            bbox_hw = int(0.5*bb_width)
            coord = (tar_[1]-bbox_hw, tar_[0]-bbox_hw, tar_[1]+bbox_hw+shift, tar_[0]+bbox_hw+shift)
            draw.rectangle(coord, outline=color)

    draw_bbox(tar_list=targets)
    draw_bbox(tar_list=targets2, shift=1, color_normal=(255, 0, 0))
    path = prefix + str(int(100000 * np.random.random_sample())) + ".png"
    img.save(path)


def metric_calc(true_pos, false_pos, false_neg):
    """ Calculation of metrics i.e precision, recall and F1-score"""
    precision_ = true_pos / (true_pos + false_pos)
    recall_ = true_pos / (true_pos + false_neg)
    f1_score = 2 * (recall_ * precision_) / (recall_ + precision_)
    return precision_, recall_, f1_score


def calc2DPrecisionRecallF1(predictions, targets, boundingBoxWidth: float, thresholdIou: float):
    """
    Calculates the precision, recall and F1-score for 2D data
    :param predictions: A list containing the 2D predictions
    :param targets: A list containing the 2D targets
    :param boundingBoxWidth: The width of the bounding box, which is used to calculate IoU
    :param thresholdIou: Suppression parameter to ignore the boxes which are less than given value.
    :return: precision, recall, f1, TP, FP and FN
    """
    tp, fn, fp = 0, 0, 0  # True positives, ...
    precision, recall, f1 = 0.0, 0.0, 0.0   # Calculate f1. But avoid div-by-zero

    for target in targets:
        targetFound = False
        for pred in predictions:
            IoU = iou(pred, target, boundingBoxWidth)
            if (IoU >= thresholdIou) and not targetFound:  # Kind of non-max-supression
                tp += 1
                targetFound = True
        if not targetFound and len(target) >= 3 and target[2] == 'normal':
            fn += 1
    for pred in predictions:
        targetFound = False
        for target in targets:
            IoU = iou(pred, target, boundingBoxWidth)
            if IoU >= thresholdIou:
                targetFound = True
                break
        if not targetFound:
            fp += 1
    if tp != 0:
        precision, recall, f1 = metric_calc(true_pos=tp, false_pos=fp, false_neg=fn)
    return precision, recall, f1, tp, fp, fn


def calcPrecisionRecallF1ForOutput(net, outputs, targets, boundingBoxWidth: float, thresholdIou: float):
    """
    Calculates the precision, recall for given model output.
    param net: Model
    param outputs:  Model output
    param targets: A list of targets
    param boundingBoxWidth: The width of the bounding box, which is used to calculate IoU
    param thresholdIou: IoU suppression factor
    :return: precision, recall, f1, TP, FP and FN
    """
    sliceAmount = len(targets)  # sliceAmount is usually equal to batch size
    tp, fn, fp = 0, 0, 0  # True positives, ...
    precision, recall, f1 = 0.0, 0.0, 0.0  # Calculate f1. But avoid div-by-zero
    for slice_ in range(sliceAmount):
        predictions = net.extractPredictions(outputs[slice_])
        p, r, f1, tp_, fp_, fn_ = calc2DPrecisionRecallF1(predictions, targets[slice_], boundingBoxWidth, thresholdIou)
        tp += tp_
        fp += fp_
        fn += fn_
    if tp != 0:
        precision, recall, f1 = metric_calc(true_pos=tp, false_pos=fp, false_neg=fn)

    return precision, recall, f1, tp, fp, fn


def iou(b1, b2, boxWidth):
    """ Intersection Over Union between two bounding boxes """
    return iou_(b1[0], b1[1], b2[0], b2[1], boxWidth)


@NUMBA.njit(cache=True)
def iou_(b1x, b1y, b2x, b2y, boxWidth):
    """
    Calculates the IoU of two Bounding-boxes
    :author: Erik Genthe
    param b1x: x-coord of the box-1
    param b1y: y-coord of the box-1
    param b2x: x-coord of the box-2
    param b2y: y-coord of the box-2
    param boxWidth: width of the box
    :return: IOU of the given two bounding boxes
    """
    def overlap(x1, w1, x2, w2):
        l1 = x1 - w1 / 2.0
        l2 = x2 - w2 / 2.0
        left = max(l1, l2)  # l1 > l2 ? l1 : l2
        r1 = x1 + w1 / 2.0
        r2 = x2 + w2 / 2.0
        right = min(r1, r2)  # r1 < r2 ? r1 : r2
        return right - left

    box1Area = boxWidth * boxWidth
    box2Area = boxWidth * boxWidth

    # Calc intersection...
    overlapX = overlap(b1x, boxWidth, b2x, boxWidth)
    overlapY = overlap(b1y, boxWidth, b2y, boxWidth)
    intersectionArea = 0
    if overlapX > 0 and overlapY > 0:
        intersectionArea = overlapX * overlapY
    # Calc union...
    unionArea = box1Area + box2Area - intersectionArea
    # Calc IoU...
    iou_ = intersectionArea / unionArea
    return iou_


def list_2_pos(file_path: str, li: list):
    """
    author: Erik Genthe
    Write a list-with 3D-Coordinates to a file
    param file_path: file path to store the resultant file
    param li: Input list, which consists of a 3D-Coordinates
    :return: None
    """
    assert isinstance(file_path, str)
    assert isinstance(li, list)
    if not file_path.lower().endswith(".pos"):
        print("Warning: In tools->list_2_pos. Expected a .pos-suffix in file_path")
    f = open(file_path, 'w')
    for l in li:
        assert len(l) == 4 or len(l) == 6                       #   REPIC_PATCH
        f.write("%f %f %f %f" % (l[2], l[1], l[0], l[3]))       #   REPIC_PATCH
        if len(l) == 6:
            f.write(" %f %f %f", (l[5], l[4], l[3]))
        f.write('\n')
    f.close()


def list2pos_wit_objectness(filepath:str, li:list):
    """
    :param filepath: Filepath to store a resultant file (To store mean objectness and objectness)
    :param li: A list with 3D-coordinates
    :return: None
    """
    assert isinstance(filepath, str)
    assert isinstance(li, list)
    if not filepath.lower().endswith(".detail"):
        print("Warning: in tools -> list_2_pos. Expected .detail-suffix in file_path")
    f = open(filepath, 'w')
    for l in li:
        f.write("%f %f %f %f" %(l[2], l[1], l[0], l[3]))
        f.write(" ")
        f.write(str(list(l[4])))
        f.write('\n')
    f.close()


def pos_2_np(file_path, only_midpoints=True):
    """
    author: Erik
    converts a 3d coordinates of a .pos file into a numpy array
    param file_path: Input .pos file path
    param only_midpoints: Mid-points
    :return: A numpy like array
    """
    result = list()
    f = open(file_path, 'r')
    for line in f.readlines():
        sp = line.strip().split()   # trim() Remove whitespace from front and back
        if len(sp) == 3:
            z, y, x = float(sp[0]), float(sp[1]), float(sp[2])
            result.append(np.array([x, y, z]))
        else:
            assert len(sp) == 6, "Expected 6 values. Got: " + str(len(sp)) + "  Line: " + line + "\nFile: " + file_path
            z = (float(sp[0]) + float(sp[3])) / 2
            y = (float(sp[1]) + float(sp[4])) / 2
            x = (float(sp[2]) + float(sp[5])) / 2
            if only_midpoints:
                result.append(np.array([x, y, z]))
            else:
                d = np.abs(float(sp[0]) - float(sp[3]))
                h = np.abs(float(sp[1]) - float(sp[4]))
                w = np.abs(float(sp[2]) - float(sp[5]))
                result.append(np.array([x, y, z, w, h, d]))
    f.close()
    return np.array(result)



def mrc_2_np(mrc_file_path):
    """
    Converts a tomogram data to a np array
    param mrc_file_path: Input .MRC filepath
    return: A numpy array
    """
    mrc_data = MRC.open(mrc_file_path)
    mrc_img = mrc_data.data

    return mrc_img
