from meghair.utils import io
from IPython import embed
from megskull.graph import FpropEnv
import nori2 as nori
import numpy as np
from meghair.utils.imgproc import imdecode
import random
import cv2
import argparse
import megbrain as mgb
from ntools.data.datapro import DataproProvider
from ntools.megtools.classification.config import DataproProviderMaker, TopNEvaluator
from ntools.megtools.classification.utils import  ScoreSet
from meghair.train.env import  OneShotEnv
def get_data():
    return DataproProviderMaker(
        config_file = 'provider_config_val.txt',
        provider_name = 'provider_cfg_val',
        entry_names = ['image_val', 'label'],
        output_names = ['data', 'label']
        )

data_func = get_data()
data_iter = data_func()
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--model', required=False, help='model, e.g., outputs/conv4_2pruned_model.save', default='outputs/pool3pruned_model.save')
parser.add_argument('-t', '--test', required=False, help='test on few examples', default=False)
#embed()
worker_name = 'val'

with OneShotEnv(worker_name, custom_parser=parser) as env:
    a = io.load(env.args.model)
    net = a
    c = [net.loss_visitor.all_oprs_dict['prob_softmax']]
#env = FpropEnv()
    fprop = env.make_func_from_loss_var(net.loss_visitor.all_oprs[0], "val", train_state=False, enforce_var_shape=False)
    fprop.compile(c)

    print("build fpropi func: Done")

    batch_num = 0

    test_scores = ScoreSet()
    N = 50000
    if env.args.test:
        N = 500
    for k in range(N//100):
        batch_num+=1
        print('Batch #',batch_num,':')
        data = next(data_iter)
        d = data[data_iter.output_names[0]]
        gt = data[data_iter.output_names[-1]]
        d[:,0,:,:]-=103.939
        d[:,1,:,:]-=116.779
        d[:,2,:,:]-=123.68
        print("load data: Done")
        ans = fprop(data = d,label = gt)[0]
        output_gt_pair = [ans, gt]
        evaluators = [
                ('Top-1 err', TopNEvaluator(1)),
                ('Top-5 err', TopNEvaluator(5))]
        scores = [(item[0], item[1](output_gt_pair)) for item in evaluators]
        # save results
        test_scores.append(scores)

# output overall results
    scores_str = "TOTAL Score: {}".format(test_scores.format())

    result = test_scores.get_result()
    return_scores = {'Top-1 err':None, 'Top-5 err': None}
    for key, val in result:
        if key in return_scores:
            return_scores[key] = val
    print(return_scores)

