from ntools.megtools.classification.config import DataproProviderMaker, TopNEvaluator
from ntools.megtools.classification.utils import ScoreSet
from meghair.train.env import TrainingEnv, Action
import argparse
from meghair.utils.io import load_network
import numpy as np

from megskull.graph import FpropEnv
from .utils import underline, OK, FAIL, space, CHECK_EQ, shell, redprint, Timer
from sklearn.linear_model import Lasso,LinearRegression, MultiTaskLasso
from .decompose import relu, dictionary, rel_error
from ntools.data.datapro import DataproProvider



class brain_net():
    def __init__(self, model_file):
        self.net = load_network(model_file)
        self.N = 128
        self.batch_size = 64
        self.new_net = load_network(model_file)
        print("finished loading weights")
        self.oprs_dict = self.net.loss_visitor.all_oprs_dict
        self.new_oprs_dict = self.new_net.loss_visitor.all_oprs_dict
        #print(self.oprs_dict)
        self.convs = self.get_convs_from_net()
        print(self.convs)
        c = []
        env = FpropEnv()
        for layers in self.net:
            c.append(env.get_mgbvar(layers))
        self.fprop1 = env.comp_graph.compile_outonly(c)

	data_func = self.get_data()
	self.data_iter = data_func()

    def get_data():
	return DataproProviderMaker(
	    config_file = 'provider_config_val.txt',
	    provider_name = 'provider_cfg_val',
	    entry_names = ['image_val', 'label'],
	    output_names = ['data', 'label']
	    )

    def get_data_batch():
        data = next(self.data_iter)
        d = data[data_iter.output_names[0]]
        gt = data[data_iter.output_names[-1]]
        d[:,0,:,:]-=103.939
        d[:,1,:,:]-=116.779
        d[:,2,:,:]-=123.68
        return d, gt


    def get_convs_from_net(self):
        convs = []
        for i in self.oprs_dict:
            if i.startswith('conv'):
                cand = i[:-2]
                if cand not in convs:
                    convs.append(cand)
        return convs

    def param_data(self, conv):
        return self.oprs_dict[conv+':W'].get_value()

    def param_b_data(self, conv):
        return self.oprs_dict[conv+':b'].get_value()
    def param_shape(self, conv):
        return self.oprs_dict[conv+':W'].get_value.shape
    def get_layer_id(self, conv):
        i = 0
        while (self.fprop1.outputs[i].name != conv):
            i += 1
        return i

    def dictionary_kernel(self, X_name, d_prime, Y_name, DEBUG = 0):
        """ channel pruning algorithm wrapper
        X_name: the conv layer to prune
        d_prime: number of preserving channels (c' in paper), the speed-up ratio = d_prime / number of channels
        Y_name: the next conv layer (For later removing of corresponding pruned weights)
        """

        X, Y = self.extract_XY(X_name, Y_name) # extract_XY(conv, convnext)
        newX = np.rollaxis(X.reshape((-1, h, w, X.shape[1])), 3, 1).copy()

        W2 = self.param_data(Y_name)
        if DEBUG: print("net.dictionary_kernel: dcfgs.ls is not gd or there is no MemoryData -by Mario")
        Y = Y - self.param_b_data(Y_name) # compute the difference between what the extracted feature and the biases of the next layer ??? -by Mario
        if DEBUG: print("gtY is only defined in this if-condition, but it is required bellow --> this condition is always be true?")

        newX = relu(newX)

        print("rMSE", rel_error(newX.reshape((newX.shape[0],-1)).dot(W2.reshape((W2.shape[0],-1)).T), gtY))
        # performe the lasso regression -by Mario
        outputs = dictionary(newX, W2, Y, rank=d_prime, B2=self.param_b_data(Y_name))
        return outputs

    def R3(self): # TODO: Delete VH and ITQ from R3 to eliminate spatial and channel factorization (tried but failed ㅜㅜ) -by Mario
        print("entered R3!!!!")
        oprs_dict = ori_net.loss_visitor.all_oprs_dict
        self.usexyz()
        speed_ratio = 3.
        prefix = str(int(speed_ratio)+1)+'x'
        DEBUG = True
        convs= self.convs
        end = 5 # TODO: Consider passing a flag to create this dictionaries for other models (passign arguments to the paserser maybe?) -by Mario
        alldic = ['conv%d_1' % i for i in range(1,end)] + ['conv%d_2' % i for i in range(3, end)]
        pooldic = {'conv1_2':'pool1', 'conv2_2':'pool2'}#, 'conv3_3']
        rankdic = {'conv1_1': 17,
                   'conv1_2': 17,
                   'conv2_1': 37,
                   'conv2_2': 47,
                   'conv3_1': 83,
                   'conv3_2': 89,
                   'conv3_3': 106,
                   'conv4_1': 175,
                   'conv4_2': 192,
                   'conv4_3': 227,
                   'conv5_1': 398,
                   'conv5_2': 390,
                   'conv5_3': 379}

        for i in rankdic:
            if 'conv5' in i:
                continue # the break-statemet was giving a bug, so changed it to continue-statement -by Mario
            rankdic[i] = int(rankdic[i] * 4. / speed_ratio)
        c_ratio = 1.15

        t = Timer()

        for conv, convnext in zip(convs[1:], convs[2:]+['pool5']): # note that we exclude the first conv, conv1_1 contributes little computation -by Mario
            W_shape = self.param_shape(conv)
            d_c = int(W_shape[0] / c_ratio)
            rank = rankdic[conv]
            d_prime = rank
            if d_c < rank: d_c = rank
            print("channel pruning")
            '''channel pruning'''
            if (conv in alldic or conv in pooldic) and (convnext in self.convs):
                t.tic()
                #if conv.startswith('conv4'): #what is this -by Mario
                #    c_ratio = 1.5
                if conv in pooldic:
                    X_name = pool_dict[X_name]
                else:
                    X_name = conv
                #idxs, array of booleans that indicates which feature maps(?) are elimated
                # newX: N c h w (BatchSize, channels, h, w), W2: n c h w (out_channels, in_channels, fitler_h, filter_w)
                idxs, W2, B2 = self.dictionary_kernel(X_name, d_c, convnext)
                # W2
                W_new = self.param_data(convnext)
                W_new[:, ~idxs, ...] = 0
                W_new[:, idxs, ...] = W2.copy()
                self.new_oprs_dict[convnext+':W'].set_value(W_new)
                self.new_oprs_dict[convnext+':b'].set_value(B2)

                # W1 #TODO: For channel pruning only, we should handle the origial conv layers (not the _H or _P layers)
                     # This section of code must be addapted

                t.toc('channel_pruning')
            print("channel pruning finished")
        new_pt = self.save_pt(prefix=prefix)
        return new_pt

    def extract_XY(self, X_name, Y_name, DEBUG = False):
        """
        given two conv layers, extract X (n, c0, ks, ks), given extracted Y(n, c1, 1, 1)
        NOTE only support: conv(X) relu conv(Y)

        Return:
            X feats of size: N C h w
        """
        batch_size = self.batch_size
        N = self.N

        c = []
        env = FpropEnv()
        for layers in self.new_net:
            c.append(env.get_mgbvar(layers))
        fprop2 = env.comp_graph.compile_outonly(c)

        k2 = self.get_layer_id(X_name)
        k1 = self.get_layer_id(Y_name)

        for i in range(N//batch_size):
            data, label = self.get_data_batch
            fprop2_out = fprop2(data = data, label = label)[k2]




        return X, Y
    def prune(self):
        print("Pruning model....")
