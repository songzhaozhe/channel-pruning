from ntools.data.datapro import DataproProvider
from ntools.megtools.classification.config import DataproProviderMaker, TopNEvaluator
from ntools.megtools.classification.utils import ScoreSet
from meghair.train.env import TrainingEnv, Action
import argparse
from meghair.utils.io import load_network
import numpy as np
from net import brain_net

model_file = 'vgg16.brainmodel'


def prune_vgg():
    net = brain_net(model_file)
    #net.prune()




def main():
    prune_vgg()

main()
