"""
Basis: https://github.com/BobLiu20/YOLOv3_PyTorch/blob/master/nets/model_main.py
Modified yolo-network.
Uses darknet-backbone. Has less embedding-layers than official yolo.
"""
from collections import OrderedDict
from numba import njit
import numpy as NP
import torch as T
import torch.nn as NN
import modules.darknet as DARKNET
from modules.darknet import Mish

OUT_FILTER = 3      # x, y, objectness :)
OUT_BIAS_INIT_VALUE = 0.0   # Initialize Objectness-bias in a such a way, that there are not too many FP from the beginnning on.
IN_OUT_DIVISOR = 8
OBJECTNESS_THRESHOLD = 0.5
TINY = True


class PickyoloNet(NN.Module):
    def __init__(self, small_network, dropout_rate=0.0, use_batchnorm=True):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        self.small_network = small_network
        self.inputSize = None
        #  backbone
        if small_network:
            self.backbone = DARKNET.darknet21(dropout_rate, use_batchnorm, TINY)
        else:
            self.backbone = DARKNET.darknet53(dropout_rate, use_batchnorm, TINY)

        _out_filters = self.backbone.layers_out_filters

        if TINY:
            D = 2
        else:
            D = 1
        #  embedding0
        self.embedding0 = self._make_embedding([512//D, 1024//D], _out_filters[-1], OUT_FILTER)
        #  embedding1
        self.embedding1_cbl = self._make_cbl(512//D, 256//D, 1)
        self.embedding1 = self._make_embedding([256//D, 512//D], _out_filters[-2] + 256//D, OUT_FILTER)
        #  embedding2
        self.embedding2_cbl = self._make_cbl(256//D, 128//D, 1)
        self.embedding2 = self._make_embedding([128//D, 256//D], _out_filters[-3] + 128//D, OUT_FILTER)

        self.logisticActivation = NN.Sigmoid()  # Von Erik

    def _make_cbl(self, _in, _out, ks):
        """cbl = conv + batch_norm + activation """
        pad = (ks - 1) // 2 if ks else 0
        l = []
        l.append(("conv", NN.Conv2d(_in, _out, kernel_size=ks, stride=1, padding=pad, bias=False)))
        if self.use_batchnorm:
            l.append(("bn", NN.BatchNorm2d(_out)))
        if self.dropout_rate > 0.0:
            l.append(("dp", NN.Dropout2d(self.dropout_rate)))
        l.append(("activation", Mish()))
        return NN.Sequential(OrderedDict(l))

    def _make_embedding(self, filters_list, in_filters, out_filter):
        m = NN.ModuleList([
            self._make_cbl(in_filters, filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            ])
        m.add_module("conv_out", NN.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True))
        # Initialize Objectness-bias in a such a way, that there are not too many FP from the beginnning on.
        m.conv_out.bias.data[2] = OUT_BIAS_INIT_VALUE
        return m

    def forward(self, x):
        # Add Padding first, so that the inputData has length and with that are a multiple of 8...
        originalInSize = (x.shape[-2], x.shape[-1])
        div = IN_OUT_DIVISOR
        padding = (0, 0 if originalInSize[1] % div == 0 else div - originalInSize[1] % div,
                   0, 0 if originalInSize[0] % div == 0 else div - originalInSize[0] % div,)
        x = NN.functional.pad(x, padding, "constant", 0)
        self.inputSize = (x.shape[-2], x.shape[-1])
        assert self.inputSize[0] % IN_OUT_DIVISOR == 0
        assert self.inputSize[1] % IN_OUT_DIVISOR == 0

        def _branch(_embedding, _in):
            out_branch = None
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 4:
                    out_branch = _in
            return _in, out_branch

        #  backbone
        x2, x1, x0 = self.backbone(x)
        #  yolo branch 0
        out0, out0_branch = _branch(self.embedding0, x0)
        #  yolo branch 1
        x1_in = self.embedding1_cbl(out0_branch)
        upShape1 = x1.shape[-2:]
        x1_in = NN.functional.interpolate(x1_in, size=upShape1)
        x1_in = T.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.embedding1, x1_in)
        #  yolo branch 2
        x2_in = self.embedding2_cbl(out1_branch)
        upShape2 = x2.shape[-2:]
        x2_in = NN.functional.interpolate(x2_in, size=upShape2)
        x2_in = T.cat([x2_in, x2], 1)
        out2, out2_branch = _branch(self.embedding2, x2_in)
        out = out2
        out = self.logisticActivation(out)
        return out

    def extractPredictions(self, netOutput):
        """ See extractPredictions() - Function """
        return extractPredictions(netOutput, self.inputSize)


def calcQuotOfInputToOutput():
    # Assert, that IN_OUT_DIVISOR is set correctly.
    # If the networktopology gets changed, IN_OUT_DIVISOR might need to be adjusted.
    # Calculate quotient of: InputWidth / OutputWidth of the Network.
    dummyData = T.rand(1, 1, 128, 128)
    net = PickyoloNet(small_network=True)
    out = net(dummyData)
    assert IN_OUT_DIVISOR == dummyData.shape[-2] / out.shape[-2]
    assert IN_OUT_DIVISOR == dummyData.shape[-1] / out.shape[-1]
    del dummyData
    del net
    del out


calcQuotOfInputToOutput()


def extractPredictions(netOutput, inputSize:tuple, objectness_threshold:float=OBJECTNESS_THRESHOLD):    #   REPIC_PATCH
    """ Takes the net-output as input, interprets it, and extracts
        the predictions made by the network. """
    netOutput = NP.array(netOutput)  # Numba can't deal with pytorch-tensors.
    return _wrappedExtractPredictions(netOutput, inputSize, objectness_threshold)                       #   REPIC_PATCH


@njit(cache=True)
def _wrappedExtractPredictions(netOutput, inputSize:tuple, objectness_threshold:float):                 #   REPIC_PATCH
    """ Numba cant deal with pytorch tensors. That is why this wrapper is needed. """

    div = IN_OUT_DIVISOR
    inputSize = (inputSize[0] + (0 if inputSize[0] % div == 0 else div - inputSize[0] % div),
                 inputSize[1] + (0 if inputSize[1] % div == 0 else div - inputSize[1] % div),)
    assert inputSize[0] % 8 == 0
    assert inputSize[1] % 8 == 0

    shape = netOutput.shape
    gridW = shape[1]
    gridH = shape[2]
    out = []
    # TODO Eventuell performance-boost durch NP.reduce() möglich.
    for cellX in range(gridW):
        for cellY in range(gridH):
            if netOutput[2][cellX][cellY] > objectness_threshold:                                      #   REPIC_PATCH
                o = netOutput[:, cellX, cellY]
                x = (cellX + o[0]) / gridW
                y = (cellY + o[1]) / gridH
                x = int(x * (inputSize[0] - 1))
                y = int(y * (inputSize[1] - 1))
                out.append((x, y, netOutput[2][cellX][cellY]))                                          #   REPIC_PATCH
                # out.append((x, y, o[2]))
    return out
