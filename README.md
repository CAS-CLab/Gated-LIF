# GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks

This project is a PyTorch implementation for

> **GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks**
>
> Xingting Yao, Fanrong Li, Zitao Mo, Jian Cheng
>
> NeurIPS 2022 Poster & Spotlight Presentation

<img src="https://github.com/Ikarosy/Gated-LIF/blob/master/IMG/GLIF.png" width="1000px">



## Getting Started

### Requirements

The main requirements of this project are as follows:
* Python 3.8.8
* PyTorch == 1.10.0+cu113
* Torchvision == 0.11.1+cu113
* CUDA 11.3
* SpikingJelly == 0.0.0.0.10



### Trained Models for Static Datasets: CIFAR, ImageNet

Our trained models can be found in `Gated-LIF/trained models`.  Download and place them in any folder you would like.  For example, `/home/GLIF_models`.



### Evaluate Trained GLIF SNNs on Static Datasets: CIFAR, ImageNet

The following are the python commands to run the python script `train.py`. We recommend using **Absolute Paths** to clarify the required Directories, and please make sure to change the current working directory to this project, i.e., `$pwd>>.../Gated-LIF`. 

Note that we utilized a single GPU for evaluations.
```python
# CIFAR-10
## Resnet-18
CUDA_VISIBLE_DEVICES=[GPU-ID] python -u train.py --modeltag [TRAINED-MODEL-FILENAME] --soft-mode --eval --eval-resume [PATH-TO-TRAINED-MODEL-FOLDER] --stand18 --channel-wise --t [TIMESTEP] --dataset-path [PATH-TO-DATASET] > evaluation.log
## Resnet-19
CUDA_VISIBLE_DEVICES=[GPU-ID] python -u train.py --modeltag [TRAINED-MODEL-FILENAME] --soft-mode --eval --eval-resume [PATH-TO-TRAINED-MODEL-FOLDER] --channel-wise --t [TIMESTEP] --dataset-path [PATH-TO-DATASET] > evaluation.log
# CIFAR-100
## Resnet-18
CUDA_VISIBLE_DEVICES=[GPU-ID] python -u train.py --modeltag [TRAINED-MODEL-FILENAME] --soft-mode --eval --eval-resume [PATH-TO-TRAINED-MODEL-FOLDER] --stand18 --channel-wise --t [TIMESTEP] --dataset-path [PATH-TO-DATASET] --cifar100 > evaluation.log
## Resnet-19
CUDA_VISIBLE_DEVICES=[GPU-ID] python -u train.py --modeltag [TRAINED-MODEL-FILENAME] --soft-mode --eval --eval-resume [PATH-TO-TRAINED-MODEL-FOLDER] --channel-wise --t [TIMESTEP] --dataset-path [PATH-TO-DATASET] --cifar100 > evaluation.log
# ImageNet
## ResNet-18MS
CUDA_VISIBLE_DEVICES=[GPU-ID] python -u train.py --modeltag [TRAINED-MODEL-FILENAME] --soft-mode --eval --eval-resume [PATH-TO-TRAINED-MODEL-FOLDER] --MS18 --channel-wise --t [TIMESTEP] --train-dir [PATH-TO-IMAGENET-TRAININGSET] --val-dir [PATH-TO-IMAGENET-VALIDATIONSET] --imagenet > evaluation.log
```
Specifically, `[TRAINED-MODEL-FILENAME]` refers to the filename of the .tar file, e.g., ''resCifar18stand-CIFAR10-step6-CW.pth.tar''. `[PATH-TO-TRAINED-MODEL-FOLDER]` refers to the folder that contains trained models, e.g., `/home/GLIF_models`. `TIMESTEP` refers to the length of the time window of the model, e.g., the timestep of the model ''resCifar18stand-CIFAR10-step6-CW.pth.tar'' is 6.

Evaluation results are printed in evaluation.log.





### Train GLIF SNNs on Static Datasets

```python
# CIFAR-10
## Resnet-18
CUDA_VISIBLE_DEVICES=[GPU-ID] python -u train.py --epoch 200 --batch-size 64 --learning-rate 0.1 --modeltag  [CHECKPOINT-FILENAME] --soft-mode --stand18 --channel-wise --randomgate --tunable-lif --t [TIMESTEP] --dataset-path [PATH-TO-DATASET] > train.log
# CIFAR-100
## Resnet-18
CUDA_VISIBLE_DEVICES=[GPU-ID] python -u train.py --epoch 200 --batch-size 64 --learning-rate 0.1 --modeltag [CHECKPOINT-FILENAME] --soft-mode --stand18 --channel-wise --randomgate --tunable-lif --t [TIMESTEP] --dataset-path [PATH-TO-DATASET] --cifar100> train.log
# ImageNet (distributed computation on multi-GPUs)
## ResNet-18MS
CUDA_VISIBLE_DEVICES=[GPU-IDs] python -m torch.distributed.run --master_port [PORT-ID] --nproc_per_node [NUMBER-OF-GPUs] train.py --epoch 150 --batch-size 50 --learning-rate 0.1 --modeltag [CHECKPOINT-FILENAME] --soft-mode --MS18 --channel-wise --randomgate --tunable-lif --t [TIMESTEP] --train-dir [PATH-TO-IMAGENET-TRAININGSET] --val-dir [PATH-TO-IMAGENET-VALIDATIONSET] --imagenet> train.log
```
Training details are printed in train.log. Checkpoints are stored in `./raw/models`.  Model options and training hyperparameters are configurable with different commands. Those commands and their descriptions can be found in `.../Gated-LIF/train.py` from line 22 to line 71.



### Train GLIF SNNs on the Dynamic Dataset: CIFAR10-DVS

We plug GLIF into an open-source project for CIFAR10-DVS, which is [SEW-PLIF-CIFAR10-DVS](https://github.com/fangwei123456/Spike-Element-Wise-ResNet/tree/main/cifar10dvs).  

The codes, trained models, and training logs for CIFAR10-DVS are saved in the file `.../Gated-LIF/cifar10dvs`.  The following is the python command that we use to train a GLIF-based 7B-wideNet:

 ```python
 #	CIFAR10-DVS
 ## 7B-wideNet
 CUDA_VISIBLE_DEVICES=6 python ./cifar10dvs/train.py --dsr_da -amp -out_dir ./logs -model SEWResNet_GLIF_dsr -cnf ADD -device cuda:0 -dts_cache /mnt/lustre/GPU8/home/usr/dvs_datasets/DVSCIFAR10/cifar10dvs_cache_SEW -epochs 200 -T_max 64 -T 16 -data_dir /mnt/lustre/GPU8/home/usr/dvs_datasets/DVSCIFAR10 -lr 0.01 -b 32 > True_widePLIF7B_GLIF-T_16-anneal-dsr-epoch_200.log
 ```

In *.../Gated-LIF/cifar10dvs*, we add some different models, including GLIF-based models, to the original [*cifar10dvs*](https://github.com/fangwei123456/Spike-Element-Wise-ResNet/tree/main/cifar10dvs).  You can easily find their model names from line 144 to line 128 in `.../Gated-LIF/cifar10dvs/train.py`.   



## Main Results of GLIF SNNs

<div style="text-align:center">
<table>
  <tr>
    <th><center>Model</center></th>
    <th><center>TimeStep</center></th>
    <th><center>CIFAR10 Top-1(%)</center></th>
    <th><center>CIFAR100 Top-1(%)</center></th>
  </tr>
  <tr>
    <td rowspan="3"><center>ResNet-18</center></td>
    <td ><center>2</center></td>
    <td ><center>94.19</center></td>
    <td ><center>74.77</center></td>
  </tr>
  <tr>
      <td ><center>4</center></td>
    <td ><center>94.75</center></td>
    <td ><center>76.50</center></td>
  </tr>
    <tr>
      <td ><center>6</center></td>
    <td ><center>95.09</center></td>
    <td ><center>77.49</center></td>
  </tr>
    <tr>
    <td rowspan="3"><center>ResNet-19</center></td>
    <td ><center>2</center></td>
    <td ><center>94.56</center></td>
    <td ><center>75.60</center></td>
  </tr>
  <tr>
     <td ><center>4</center></td>
    <td ><center>94.95</center></td>
    <td ><center>77.22</center></td>
  </tr>
    <tr>
      <td ><center>6</center></td>
    <td ><center>95.14</center></td>
    <td ><center>77.42</center></td>
     </tr>
</table>
</div>
<div style="text-align:center">
<table >
  <tr>
    <th><center>Model</center></th>
    <th><center>Dataset</center></th>
    <th><center>TimeStep</center></th>
    <th><center>Top-1(%)</center></th>
  </tr>
      <tr>
     <td><center>7B-wideNet</center></td>
    <td ><center>CIFAR10-DVS</center></td>
    <td ><center>16</center></td>
    <td ><center>78.10</center></td>
  </tr>
      <tr>
      <td ><center>ResNet-18MS</center></td>
    <td ><center>ImageNet</center></td>
    <td ><center>6</center></td>
    <td ><center>68.10</center></td>
  </tr>
</table>
</div>



P.S. the CIFAR10-DVS result is 1.3% higher than reported in the *openreview* discussion.  Because we fix a minor bug in the `.../Gated-LIF/cifar10dvs/smodels`. The fixed script, new training logs, and to-date trained models have been updated or added by 2022/11/3, which should work well and match the results in the above list. The paper on the *openreview* has already been fixed and corrected.



## Further Exploration & Future Expectation

1. In the script `.../Gated-LIF/train.py`, we retain some useful python commands to reproduce our ablation studies. Anyone who reads the parser descriptions from line 22 to 71 should easily understand how to use them.
2. Furthermore, we retain some codes to support the experiments of unstudied GLIF-based variants and some tricks.  For example, making all the gating factors learnable but keeping binary, referred to as 'hard mode', still improves performance compared to some LIF-based SNNs.  Unlike the proposed GLIF method in the paper, the 'hard mode' should require the same computation overhead as the normal LIFs, increasing the heterogeneity of SNNs. (Experimental results of 'Hard Mode' GLIF will be revealed as extended studies in our in-progress works.)
3. The distributions of learned parameters are very interesting, as we visualized them in the paper.  The initially identical parameters learn into different bell-shaped distributions layer-wisely.  This may shed light on some interesting connections between DNNs and the hierarchical structures of brains.
4. Since GLIF offers more tunable parameters than LIF, extending it into the frameworks of the ANN2SNN conversion should be interesting because the recent trend of ANN2SNN is figuring out better parameter mapping from ANNs to SNNs to improve the performance of the converted SNNs.  Hopefully, this could pave a new path to that field if we can find the parameter mapping from ANNs to GLIF-based SNNs.



## Citation

Please cite this paper using the following BibTeX entry if you find this work useful for your research.

```tex
@inproceedings{
yao2022glif,
title={{GLIF}: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks},
author={Xingting Yao and Fanrong Li and Zitao Mo and Jian Cheng},
booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
year={2022},
url={https://openreview.net/forum?id=UmFSx2c4ubT}
}
```



## Contact

Please feel free to contact us if you need any further information.

>  yaoxingting2020@ia.ac.cn

