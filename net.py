from ntools.megtools.classification.config import DataproProviderMaker, TopNEvaluator
from ntools.megtools.classification.utils import ScoreSet
from meghair.train.env import TrainingEnv, Action
import argparse
from meghair.utils.io import load_network
from meghair.utils import io
import numpy as np
from meghair.train.env import  OneShotEnv

from megskull.graph import FpropEnv
from lib.utils import underline, OK, FAIL, space, CHECK_EQ, shell, redprint, Timer
from sklearn.linear_model import Lasso,LinearRegression, MultiTaskLasso
from lib.decompose import relu, dictionary, rel_error
from ntools.data.datapro import DataproProvider
from ntools.megtools.classification.config import DpflowProviderMaker



class brain_net():
    def __init__(self, model_file, env):
        self.env = env
        self.net = load_network(model_file)
        self.N = 5000
        self.batch_size = 64
        self.nperimage = 10
        self.new_net = load_network(model_file)
        print("finished loading weights")
        self.oprs_dict = self.net.loss_visitor.all_oprs_dict
        self.new_oprs_dict = self.new_net.loss_visitor.all_oprs_dict

        self.convs = self.get_convs_from_net()
        print(self.convs)
        data_func = self.get_data(is_val=False)
        data_func_val = self.get_data(is_val=True)
        self.data_iter = data_func()
        self.data_iter_val = data_func_val()

    def get_data(self, is_val = False):
        if is_val:
            return DataproProviderMaker(
                config_file = 'provider_config_val.txt',
                provider_name = 'provider_cfg_val',
                entry_names = ['image_val', 'label'],
                output_names = ['data', 'label']
                )
        else:
            return DpflowProviderMaker(
                conn                = 'szz.vgg.imagenet.val',
                entry_names         = ['image', 'label'],
                output_names        = ['data', 'label'],
                descriptor          = { 'data': { 'shape': [64, 3, 224, 224] }, 'label': { 'shape': [64] } },
                buffer_size         = 16
                )

    def get_data_batch(self, is_val = False):
        #print("getting")
        if is_val == False:
            data = next(self.data_iter)
        else:
            data = next(self.data_iter_val)
        d = data[self.data_iter.output_names[0]]
        gt = data[self.data_iter.output_names[-1]]
        d[:,0,:,:]-=103.939
        d[:,1,:,:]-=116.779
        d[:,2,:,:]-=123.68
        return d, gt

    def get_convs_from_net(self):
        convs = []
        for i in self.oprs_dict:
            if i.startswith('conv') and (i.endswith(':W') or i.endswith(':b')):
                cand = i[:-2]
                if cand not in convs:
                    convs.append(cand)
        return convs

    def param_data(self, conv):
        return self.oprs_dict[conv+':W'].get_value()
    def param_b_data(self, conv):
        return self.oprs_dict[conv+':b'].get_value()
    def param_shape(self, conv):
        return self.oprs_dict[conv+':W'].get_value().shape

    def dictionary_kernel(self, X_name, d_prime, Y_name, DEBUG = 0):
        """ channel pruning algorithm wrapper
        X_name: the conv layer to prune
        d_prime: number of preserving channels (c' in paper), the speed-up ratio = d_prime / number of channels
        Y_name: the next conv layer (For later removing of corresponding pruned weights)
        """

        X, Y = self.extract_XY(X_name, Y_name) # extract_XY(conv, convnext)
        W2 = self.param_data(Y_name)
        Y = Y - self.param_b_data(Y_name) # compute the difference between what the extracted feature and the biases of the next layer ??? -by Mario
        newX = relu(X)

        print("rMSE", rel_error(newX.reshape((newX.shape[0],-1)).dot(W2.reshape((W2.shape[0],-1)).T), Y))
        # perform the lasso regression -by Mario
        outputs = dictionary(newX, W2, Y, rank=d_prime, B2=self.param_b_data(Y_name))
        print("out of dic")
        return outputs

    def R3(self):
        print("entered R3!!!!")
        oprs_dict = self.oprs_dict
        new_oprs_dict = self.new_oprs_dict
        DEBUG = True
        convs= self.convs
        end = 5
        speed_ratio = 4
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
                if conv in pooldic:
                    X_name = pooldic[conv]
                else:
                    X_name = conv
                print(X_name, convnext)
                idxs, W2, B2 = self.dictionary_kernel(X_name, d_c, convnext)
                # W2
                W_new = self.param_data(convnext)
                W_new[:, ~idxs, ...] = 0
                W_new[:, idxs, ...] = W2.copy()
                self.new_oprs_dict[convnext+':W'].set_value(W_new)
                self.new_oprs_dict[convnext+':b'].set_value(B2)

                t.toc('channel_pruning')
            print("channel pruning finished")
        self.val()
        io.dump(self.new_net, 'pruned_model.save')

    def extract_XY(self, X_name, Y_name, DEBUG = False):
        """
        given two conv layers, extract X (n, c0, ks, ks), given extracted Y(n, c1, 1, 1)
        NOTE only support: conv(X) relu conv(Y)

        Return:
            X feats of size: N C h w
            Y feats of size: N c1 1 1
        """
        print("extracting XY")
        env = self.env
        batch_size = self.batch_size
        N = self.N
        n = self.nperimage
        c = [self.new_net.loss_visitor.all_oprs_dict[X_name],  self.new_net.loss_visitor.all_oprs_dict['prob_softmax']]
        fprop2 = env.make_func_from_loss_var(self.new_net.loss_visitor.all_oprs[0], "val", train_state=False, enforce_var_shape=False)
        fprop2.compile(c)

        c2 = [self.net.loss_visitor.all_oprs_dict[Y_name], self.net.loss_visitor.all_oprs_dict['prob_softmax']]
        fprop1 = env.make_func_from_loss_var(self.new_net.loss_visitor.all_oprs[0], "val", train_state=False, enforce_var_shape=False)
        fprop1.compile(c2)

        t = Timer()
        t.tic()
        sample_d, sample_l = self.get_data_batch()
        t.toc("sample data")
        print(sample_l)
        sample_X_out = fprop2(data=sample_d, label = sample_l)[0]
        t.toc("fprop2")
        sample_Y_out = fprop1(data=sample_d, label = sample_l)[0]
        t.toc("fprop1")
        X_channels = sample_X_out.shape[1]
        width = sample_X_out.shape[2]
        height = width
        Y_channels = sample_Y_out.shape[1]

        NN = N * n
        X = np.zeros([NN, X_channels, 3, 3])
        Y = np.zeros([NN, Y_channels])
        test_scores = ScoreSet()
        for i in range(N//batch_size):
            if i % (N//batch_size/10) == 0:
                print("batch_num", i)
            #t = Timer()
            #t.tic()
            data, label = self.get_data_batch()
            #t.toc("data")
            tmp = fprop2(data = data, label = label)
            #t.toc("prop")
            X_out = tmp[0]
            ans = tmp[1]
            output_gt_pair = [ans, label]
            evaluators = [
                    ('Top-1 err', TopNEvaluator(1)),
                    ('Top-5 err', TopNEvaluator(5))]
            scores = [(item[0], item[1](output_gt_pair)) for item in evaluators]
            # save results
            test_scores.append(scores)
            Y_out = fprop1(data = data, label = label)[0]
            X_complete = np.zeros([X_out.shape[0], X_out.shape[1], X_out.shape[2]+2, X_out.shape[3]+2])
            X_complete[:,:,1:-1,1:-1] = X_out
            pos_x = np.random.randint(X_out.shape[2], size=(batch_size,n)) + 1
            pos_y = np.random.randint(X_out.shape[3], size=(batch_size,n)) + 1

            for j in range(batch_size):
                for k in range(n):
                    index = (i * batch_size + j)*10 + k
                    X[index,:,:,:] = X_complete[j,:,pos_x[j][k]-1:pos_x[j][k]+2,pos_y[j][k]-1:pos_y[j][k]+2]
                    Y[index,:] = Y_out[j,:,pos_x[j][k]-1,pos_y[j][k]-1]

        cores_str = "TOTAL Score: {}".format(test_scores.format())

        result = test_scores.get_result()
        return_scores = {'Top-1 err':None, 'Top-5 err': None}
        for key, val in result:
            if key in return_scores:
                return_scores[key] = val
        print(return_scores)

        Y.reshape([Y.shape[0],Y.shape[1],1,1])

        return X, Y

    def val(self):
        print("validating....")
        c = [self.new_net.loss_visitor.all_oprs_dict['prob_softmax']]
        #env = FpropEnv()
        env = self.env
        fprop2 = env.make_func_from_loss_var(self.new_net.loss_visitor.all_oprs[0], "val", train_state=False, enforce_var_shape=False)
        fprop2.compile(c)
        test_scores = ScoreSet()
        for i in range(500):
            if (i % 50 == 0):
                print("validated  batches", i)
            val_d, val_l = self.get_data_batch(is_val = True)
            ans = fprop2(data=val_d, label=val_l)[0]
            output_gt_pair = [ans, val_l]
            evaluators = [
                    ('Top-1 err', TopNEvaluator(1)),
                    ('Top-5 err', TopNEvaluator(5))]
            scores = [(item[0], item[1](output_gt_pair)) for item in evaluators]
            # save results
            test_scores.append(scores)
        cores_str = "TOTAL Score: {}".format(test_scores.format())

        result = test_scores.get_result()
        return_scores = {'Top-1 err':None, 'Top-5 err': None}
        for key, val in result:
            if key in return_scores:
                return_scores[key] = val
        print(return_scores)

    def prune(self):
        print("Pruning model....")
        self.R3()

def make_parser():
    parser = argparse.ArgumentParser()
    return parser


def main():
    worker_name = 'brain.test'
    parser = make_parser()
    with OneShotEnv(worker_name, custom_parser=parser) as env:
        net = brain_net('vgg16.brainmodel', env)
        #net.val()
        net.prune()


main()
