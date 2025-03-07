# GGPA (Pytorch)

This repository contains an official pytorch implementation for the following paper:
Gradient-Guided Grow-and-Prune Algorithm for Efficient Neural Network Compression(2025)
Jiaji Wua, Yuqian Duana, Qi Yua, Xijun Lianga, and Ling Jian

The code is based on [pytorch-slimming](https://github.com/foolwood/pytorch-slimming). We implemented our growth strategy and pruning strategy based on the original code, and modified the network files.  

The code mainly demonstrates the VGG19 network on the CIFAR dataset.

Regarding CIFAR data, the code will automatically create a folder in the current directory for downloading during runtime.

## Dependencies
pytorch v1.11.0, Python v3.8(ubuntu20.04), Cuda v11.3

## Channel Selection Layer
We introduce [channel selection](https://github.com/Eric-mingjie/network-slimming) layer to help the  pruning of ResNet and DenseNet. This layer is easy to implement. It stores a parameter `indexes` which is initialized to an all-1 vector. During pruning, it will set some places to 0 which correspond to the pruned channels.

## Hyperparameter Description
The 'epoch' in the terminal command represents the total number of training epochs, ‘ Net 'represents the selected network, ‘ Refine 'represents the path saved for the pruned channel, 'grow' represents the path where the model is saved after each growth.

## Baseline 

```shell
python \Baseline.py -sr --s 0.0001  --epoch 160 --net 20
python \Baseline.py -sr --s 0.0001  --epoch 160 --net 19
```
19 represents the original VGG19 network, and 20 represents the seed network of VGG19.

## Train

```shell
python \vggmain.py -sr --s 0.0001  --epoch 410 --cuttingtime 330 --percent 0.2 --growingtime 190 --net 20 --refine pruned.pth.tar --grow growing.pth.tar
```

## Retrain

```shell
python \Fine-tune.py -sr --s 0.0001  --epoch 160 --refine pruned.pth.tar
```

## Results

### CIFAR10
|  CIFAR10-Vgg19  | Baseline-S |  Baseline-B | Bn-Prune | GGPA |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |
| Top1 Accuracy (%) |  84.91   |            93.66            |        93.69        |         94.02        |
|    Parameters     |   0.32M  |            20.04M           |        2.30M        |         1.48M        |
|      Flops        |   6.5M   |            399.2M           |        195.5M       |        174.9M        |
|      Epochs       |  ------  |            -----            |         480         |          410         |

### CIFAR100
|  CIFAR100-Vgg19  | Baseline-S |  Baseline-B | Bn-Prune | GGPA |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |
| Top1 Accuracy (%) |  57.62   |            72.90            |        73.16        |         73.86        |
|    Parameters     |   0.32M  |            20.04M           |        5.00M        |         2.90M        |
|      Flops        |   6.6M   |            399.2M           |        250.5M       |        258.7M        |
|      Epochs       |  ------  |            -----            |         480         |          410         |

## Note
The data for Bn-prune comes from [Learning Efficient Convolutional Networks Through Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) (ICCV 2017).
