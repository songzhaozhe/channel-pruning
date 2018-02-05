from meghair.train.env import TrainingEnv, OneShotEnv, Action
import argparse
from lib.net import brain_net

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

if __name__ == '__main__':
	main()

main()