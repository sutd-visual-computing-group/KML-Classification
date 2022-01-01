# Revisit Multimodal Meta-Learning through the Lens of Multi-Task Learning

This project is a PyTorch implementation of the **K**ernel **M**odu**L**ation (**KML**) algorithm proposed in [Revisit Multimodal Meta-Learning through the lens of Multi-Task Learning](https://arxiv.org/pdf/2110.14202.pdf), which is published in [NeurIPS 2021](https://neurips.cc/). Please visit our [projrct page](https://miladabd.github.io/KML/) for more information.

![Figure 1_v3](https://user-images.githubusercontent.com/29326313/137575589-e8e4e88f-813f-4eed-b4ac-c05672f018b8.jpg)

**Multimodal meta-learning** is a recent problem that extends conventional few-shot meta-learning by generalizing its setup to diverse multimodal task distributions. This setup mimics how humans make use of a diverse set of prior skills to learn new skills. Previous work has achieved encouraging performance. In particular, in spite of the diversity of the multimodal tasks, previous work claims that a single meta-learner trained on a multimodal distribution can sometimes outperform multiple specialized meta-learners trained on individual unimodal distributions. The improvement is attributed to **knowledge transfer** between different modes of task distributions. However, there is no deep investigation to verify and understand the knowledge transfer between multimodal tasks. Our work makes two contributions to multimodal meta-learning. First, we propose a method to analyze and quantify knowledge transfer across different modes at a micro-level. Our quantitative, task-level analysis is inspired by the recent **transference idea** from multi-task learning. Second, inspired by hard parameter sharing in multi-task learning and a new interpretation of related work, we propose a new multimodal meta-learner that outperforms existing work by considerable margins. While the major focus is on multimodal meta-learning, our work also attempts to shed light on task interaction in conventional meta-learning.

![KML Algorithm](https://user-images.githubusercontent.com/29326313/137575826-123726c9-5414-43ad-8217-d463c356b047.jpg)

## Datasets

Run the following command to download and preprocess the datasets

`$python download.py --dataset aircraft bird cifar miniimagenet`

Or simply download it from 
[here](https://drive.google.com/file/d/1a5dfLQVBSTTLTo6QXXb5eoA6PWipVCAR/view?usp=sharing),
and put it in `data` folder.



## Getting Started

### Training

Use following commands to train the model and reproduce our results:


#### 2Mode (Omniglot & mini-ImageNet)

| Setup         | Command       |
| ------------- |:-------------|
| 5w1s          | `main.py --multimodal_few_shot omniglot miniimagenet --kml-model True --output-folder kml_2mode_5w1s`|
| 5w1s          | `main.py --multimodal_few_shot omniglot miniimagenet --kml-model True --num-samples-per-class 5 --output-folder kml_2mode_5w5s` |

#### 2Mode:new: (mini-ImageNet & FC100)

| Setup         | Command       |
| ------------- |:-------------|
| 5w1s          | `main.py --multimodal_few_shot miniimagenet FC100 --kml-model True --output-folder kml_2mode_dag_5w1s`|
| 5w1s          | `main.py --multimodal_few_shot miniimagenet FC100 --kml-model True --num-samples-per-class 5 --output-folder kml_2mode_dag_5w5s` |

#### 3Mode (Omniglot, mini-ImageNet & FC100)

| Setup         | Command       |
| ------------- |:-------------|
| 5w1s          | `main.py --multimodal_few_shot omniglot miniimagenet FC100 --kml-model True --output-folder kml_3mode_5w1s`|
| 5w1s          | `main.py --multimodal_few_shot omniglot miniimagenet FC100 --kml-model True --num-samples-per-class 5 --output-folder kml_3mode_5w5s` |

#### 2Mode (Omniglot, mini-ImageNet, FC100, CUB & Aircraft)

| Setup         | Command       |
| ------------- |:-------------|
| 5w1s          | `main.py --multimodal_few_shot omniglot miniimagenet FC100 bird aircraft --kml-model True --output-folder kml_5mode_5w1s`|
| 5w1s          | `main.py --multimodal_few_shot omniglot miniimagenet FC100 bird aircraft --kml-model True --num-samples-per-class 5 --output-folder kml_5mode_5w5s` |

### Evaluating Trained Model
In order to evaluate the trained model, load the saved checkpoint for the desired model in `--eval` mode.

For example, for 3Mode, 5w1s setup, use the follosing commend:

`main.py --multimodal_few_shot omniglot miniimagenet FC100 --kml-model True --checkpoint ./train_dir/kml_5mode_5w1s/model_gatedconv_30000.pt --eval`

## Cite the paper

If you find this useful, please cite our paper

```
@inproceedings{
abdollahzadeh2021revisit,
title={Revisit Multimodal Meta-Learning through the Lens of Multi-Task Learning},
author={Milad Abdollahzadeh and Touba Malekzadeh and Ngai-man Cheung},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://openreview.net/forum?id=V5prUHOrOP4}
}
```


## Authors
[Milad Abdollahzadeh](miladabd.github.io), [Touba Malekzadeh](https://scholar.google.com/citations?user=DgnZKiQAAAAJ&hl=en), [Ngai-Man (Man) Cheung](https://istd.sutd.edu.sg/people/faculty/ngai-man-man-cheung) 


## Refrences
[MMAML](https://github.com/shaohua0116/MMAML-Classification)

[ProtoNet](https://github.com/jakesnell/prototypical-networks)


