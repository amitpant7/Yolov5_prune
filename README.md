
# YOLOv5 Pruning Documentation

## Overview
This document provides a guide on pruning YOLOv5 models using `torch-pruning` and the `ultralytics YOLOv5` implementation. It describes the modified training process, the pruning criteria used, and the overall pruning workflow, including one-shot pruning and iterative pruning.

Model pruning helps in reducing the model size, speeding up inference, and potentially improving the deployment of models in resource-constrained environments.

## Prerequisites
Ensure you have the following libraries installed:
- `ultralytics YOLOv5` (available on GitHub)
- `torch-pruning` (available [here](https://github.com/VainF/Torch-Pruning))

Before starting, ensure your environment is set up with these libraries.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Model Pruning Workflow](#model-pruning-workflow)
- [Pruning Criteria](#pruning-criteria)
- [Pruning in Practice](#pruning-in-practice)
  - [One-shot Pruning](#one-shot-pruning)
  - [Iterative Pruning](#iterative-pruning)
- [Notebook and Technical Details](#notebook-and-technical-details)
- [Conclusion](#conclusion)

---

## Model Pruning Workflow

In this implementation, two methods are used for pruning:
1. **Iterative Pruning**: This method prunes the model step-by-step, retraining it after each pruning step. The process is executed using `prune.py`.
2. **One-shot Pruning**: This method prunes the model in one shot, followed by retraining. The process is handled by `prune_one_shot.py`.

The pruning process begins with identifying the least important weights in the model, which are then pruned according to the defined criteria.

### Key Components:
- **pruner.py**: Handles the iterative pruning and retraining process.
- **pruner_one_shot.py**: Prunes the model to the desired ratio in one shot and retrains it.
- **LAMPImportance**: The default importance criterion for pruning, which can be replaced with any other from `torch-pruning`.

## Pruning Criteria

The importance of each weight in the model is determined by the **LAMPImportance** criterion by default, which ranks the weights based on their contribution to the model's performance. Other importance criteria can be used, such as:
- Magnitude-based pruning
- Random pruning
- Gradient-based pruning

For more details, refer to the official [torch-pruning documentation](https://github.com/VainF/Torch-Pruning).

## Pruning in Practice

### Iterative Pruning
For iterative pruning, use the following command:

```bash
!python prune.py --data VOC.yaml --weights /content/best.pt --epochs 2 --cache --img 416 --project VOC --name prune_yolo
```

This command:
- Uses the `VOC.yaml` dataset.
- Loads pre-trained weights from `best.pt`.
- Sets the number of training epochs to `2`.
- Specifies an image size of `416x416` pixels.
- Saves the output to the `VOC/prune_yolo` folder.

#### Modify for Your Needs:
You can adjust parameters like `--data`, `--weights`, and `--img` to fit your dataset, model, and image size.

### One-shot Pruning
For one-shot pruning, use the following script:

```bash
!python prune_one_shot.py --data VOC.yaml --weights /content/best.pt --prune-ratio 0.5 --epochs 5
```

In this command:
- `--prune-ratio 0.5`: Prunes the model to 50% of its original size.
- `--epochs 5`: Retrains the model for 5 epochs after pruning.

## Notebook and Technical Details
A detailed walkthrough of model pruning techniques and processes can be found in the [pruning notebook](./Model_Pruning/Notebooks/pruning_yolov5_on_pascal_voc.ipynb) at `./Notebooks`. This notebook explains:
- How to use different pruning strategies.
- A comparison of one-shot and iterative pruning.
- How to evaluate the pruned model.

For more technical information on the implementation, refer to [Pruning Implementation.pdf](./Pruning%20Implementation.pdf), which provides a deep dive into the code structure and technical details of the pruning process.
