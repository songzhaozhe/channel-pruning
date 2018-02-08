# [Channel Pruning for Accelerating Very Deep Neural Networks](https://arxiv.org/abs/1707.06168)
**ICCV 2017**, by [Yihui He](http://yihui-he.github.io/), [Xiangyu Zhang](https://scholar.google.com/citations?user=yuB-cfoAAAAJ&hl=en&oi=ao) and [Jian Sun](http://jiansun.org/)

In this repository, I replicated his results in MegBrain. The original repo in Caffe is https://github.com/yihui-he/channel-pruning.

To run the code, first open DPflow: (need to modify path in provider_config_train.txt)

```bash
cd szz.vgg.imagenet.val
rlaunch --cpu=4 --gpu=0 --memory=10000 -P15 -- ~/nbin/classification/data_server provider_config_train.txt --as-uint8 image
```

Next, you can run the main function, e.g.,

```bash
python3 train.py -d 0-3
```

By default, it will prune the VGG model to a 5X one, achieving 30% top-5 error without fine-tune. It will dump the model after every layer is pruned into outputs/.

You can also test any model in outputs/ by running:

```
python3 val.py -d 0-3 --model outputs/<model-name>
```



If you want to prune other models, the 'rankdic' dict stores how many channels to preserve in every feature map. You might need to find a good configuration of 'rankdic' to get good results.