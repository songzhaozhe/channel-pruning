import numpy as np
import numpy.random as rng

import sys
import nori2
import time
import os
import gzip
import pickle
import itertools
import getpass
import logging
import argparse
import multiprocessing
from pathlib import Path

from dpflow import control, OutputPipe
from meghair.utils.misc import (
    inf_shuffled, stable_rand_seed, add_rand_seed_entropy, ProgressReporter,
)
from meghair.utils import logconf

IMAGE_SIZE = 32
MINIBATCH_SIZE = 512
LABEL_SIZE = 10
TRAIN_SIZE = 50000
TEST_SIZE = 10000
PAD_CROP = 4

logger = logconf.get_logger(__name__)


x = np.load('x.npy')
y = np.load('y.npy')
test_x = np.load('test_x.npy')
test_y = np.load('test_y.npy')

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def datafeed_addr(dataset_name):
    return getpass.getuser() + '.cifar.' + dataset_name

class Augmentor():

    @classmethod
    def distort(cls, img):
        # pad and crop settings
        trans_1 = rng.randint(0, (PAD_CROP*2))
        trans_2 = rng.randint(0, (PAD_CROP*2))
        crop_x1 = trans_1
        crop_x2 = (IMAGE_SIZE + trans_1)
        crop_y1 = trans_2
        crop_y2 = (IMAGE_SIZE + trans_2)

        # flip left-right choice
        flip_lr = rng.randint(0,2)

        # set empty copy to hold augmented images so that we don't overwrite
        img_aug = np.copy(img)
        mean = [125, 123, 114]
        for k in range(img_aug.shape[0]):
            # pad and crop images
            img_pad = np.pad(img_aug[k], pad_width=((PAD_CROP,PAD_CROP), (PAD_CROP,PAD_CROP)), mode='constant', constant_values = mean[k])
            img_aug[k] = img_pad[crop_x1:crop_x2, crop_y1:crop_y2]

            # flip left-right if chosen
            if flip_lr == 1:
                img_aug[k] = np.fliplr(img_aug[k])
        return img_aug

    @classmethod
    def process(cls, img):
        return cls.distort(img)

class DataFetcher():
    def __init__(self, is_train):

        if is_train:
            total_len = TRAIN_SIZE
            def indexer():
                while True:
                    yield rng.randint(0, total_len)
        else:
            total_len = TEST_SIZE
            def indexer():
                while True:
                    for i in range(total_len):
                        yield i

        self._idx_iter = iter(indexer())

    def load(self, is_train):
        if is_train:
            index = next(self._idx_iter)
            img = x[index]
            label = y[index]
            return img, label
        else:
            index = next(self._idx_iter)
            img = test_x[index]
            # label = index
            label = test_y[index]
            return img, label

# we should put cifar-10-batches-py folder under the same folder
def train_dataset(verbose=False, batchSize=256):
    fetcher = DataFetcher(True)
    while True:
        X = np.ndarray((batchSize, 3, IMAGE_SIZE, IMAGE_SIZE), dtype='float32')
        Y = np.zeros((batchSize), dtype='float32')
        for i in range(batchSize):
            img, label = fetcher.load(True)
            img = Augmentor.process(img)
            X[i] = img
            Y[i] = label
        yield {'img': X, 'label': Y}

def val_dataset(verbose=False, batchSize=1):
    fetcher = DataFetcher(False)
    while True:
        X = np.ndarray((batchSize, 3, IMAGE_SIZE, IMAGE_SIZE), dtype='float32')
        Y = np.zeros((batchSize), dtype='float32')
        for i in range(batchSize):
            img, label = fetcher.load(False)
            X[i] = img
            Y[i] = label
        yield {'img': X, 'label': Y}

def worker():

    #add_rand_seed_entropy(worker_id)
    np.random.seed(int(time.time()*100000%1000000))

    fetcher = DataFetcher(is_train = True)

    addr = datafeed_addr('train')
    q = OutputPipe(addr, buffer_size=100)

    with control(io=[q]):
        logger.info('started worker')
        while True:
            img, label = fetcher.load(True)
            img_aug = Augmentor.process(img)
            # msgdata cannot serialize float32
            label = int(label)
            img_aug = img_aug.astype(np.uint8)
            data = {'img':img_aug, 'label': [label]}
            q.put_pyobj(data)
    #        print('putted')
            sys.stdout.flush()

def main():

    global x
    global y
    global test_x
    global test_y
    print('... preprocessing data')

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-j', '--jobs', default=10,
        help='number of concurrent subprocesses for training',
    )
    parser.add_argument(
        '-v', '--verbose', action="store_true",
        help="print DEBUG info",
    )
    args = parser.parse_args()
    print('start worker')
    worker()

if __name__ == '__main__':
    main()
