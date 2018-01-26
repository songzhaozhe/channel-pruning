
import megbrain as mgb
import megskull as mgsk

from megskull.graph import FpropEnv
from megskull.opr.compatible.caffepool import CaffePooling2D
from megskull.opr.arith import ReLU
from megskull.opr.all import (
        DataProvider, Conv2D, Pooling2D, FullyConnected,
        Softmax, Dropout, BatchNormalization, CrossEntropyLoss,
        ElementwiseAffine, WarpPerspective, WarpPerspectiveWeightProducer,
        WeightDecay)
from megskull.network import RawNetworkBuilder


import sys
sys.setrecursionlimit(10000)


def create_conv_relu(conv_name, f_in, ksize, stride, pad, num_outputs, has_relu=True):
    f = Conv2D(conv_name, f_in, kernel_shape=ksize, stride=stride, padding=pad, output_nr_channel=num_outputs,
            nonlinearity=mgsk.opr.helper.elemwise_trans.Identity())
    if has_relu:
        f = ReLU(f)
    return f

def make_network():
    batch_size = 200
    img_size = 224

    data = DataProvider("data", shape=(batch_size, 3, img_size, img_size))
    label = DataProvider("label", shape=(batch_size, ))

    f = create_conv_relu("conv1_1", data, ksize=3, stride=1, pad=1, num_outputs=64)
    f = create_conv_relu("conv1_2", f, ksize=3, stride=1, pad=1, num_outputs=64)
    f = CaffePooling2D("pool1", f, window=2, stride=2, padding=0, mode="MAX")

    f = create_conv_relu("conv2_1", f, ksize=3, stride=1, pad=1, num_outputs=128)
    f = create_conv_relu("conv2_2", f, ksize=3, stride=1, pad=1, num_outputs=128)
    f = CaffePooling2D("pool2", f, window=2, stride=2, padding=0, mode="MAX")

    f = create_conv_relu("conv3_1", f, ksize=3, stride=1, pad=1, num_outputs=256)
    f = create_conv_relu("conv3_2", f, ksize=3, stride=1, pad=1, num_outputs=256)
    f = create_conv_relu("conv3_3", f, ksize=3, stride=1, pad=1, num_outputs=256)
    f = CaffePooling2D("pool3", f, window=2, stride=2, padding=0, mode="MAX")

    f = create_conv_relu("conv4_1", f, ksize=3, stride=1, pad=1, num_outputs=512)
    f = create_conv_relu("conv4_2", f, ksize=3, stride=1, pad=1, num_outputs=512)
    f = create_conv_relu("conv4_3", f, ksize=3, stride=1, pad=1, num_outputs=512)
    f = CaffePooling2D("pool4", f, window=2, stride=2, padding=0, mode="MAX")

    f = create_conv_relu("conv5_1", f, ksize=3, stride=1, pad=1, num_outputs=512)
    f = create_conv_relu("conv5_2", f, ksize=3, stride=1, pad=1, num_outputs=512)
    f = create_conv_relu("conv5_3", f, ksize=3, stride=1, pad=1, num_outputs=512)
    f = CaffePooling2D("pool5", f, window=2, stride=2, padding=0, mode="MAX")

    f = FullyConnected("fc6", f, output_dim=4096,
            nonlinearity=mgsk.opr.helper.elemwise_trans.Identity())
    f = ReLU(f)

    f = FullyConnected("fc7", f, output_dim=4096,
            nonlinearity=mgsk.opr.helper.elemwise_trans.Identity())
    f = ReLU(f)

    f = FullyConnected("fc8", f, output_dim=1000,
            nonlinearity=mgsk.opr.helper.elemwise_trans.Identity())

    f = Softmax("cls_softmax", f)

    net = RawNetworkBuilder(inputs=[data, label], outputs=[f], loss=CrossEntropyLoss(f, label))
    return net

