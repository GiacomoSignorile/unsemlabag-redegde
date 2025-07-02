<div align="center">
    <h1>Unsupervised Semantic Label Generation in Agricultural Fields</h1>
    <p><em>An Adapted Framework for the WeedMap Dataset</em></p>
    <br />
    <img src='pics/overview.png' width="700">
    <br/>
    <a href=https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/roggiolani2025frai.pdf>Original Paper</a>
    <span>  •  </span>
    <a href=https://github.com/PRBonn/unsemlabag/issues>Contact Us</a>
    <br />
    <br />
</div>

This repository contains the code for "Unsupervised Semantic Label Generation in Agricultural Fields" \[[1](#citation)\], specifically adapted to run experiments on the **WeedMap** dataset \[[2](#citation)\].

The framework simulates a drone's flight over a field orthomosaic, using computer vision heuristics to generate pseudo-labels, which are then used to train an evidential deep learning network for semantic segmentation.

---

## 1. Setup

### Environment

We recommend using Docker for a reproducible environment. You may need to [install the NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to enable GPU access.

Build the Docker image:
```bash
make build
```

### Download the WeedMap Dataset

This project is configured to work with the RedEdge subset of the WeedMap dataset.

 you can manually download the [RedEdge data](http://robotics.ethz.ch/~asl-datasets/2018-weedMap-dataset-release/Orthomosaic/RedEdge.zip) and extract it to a `samples/weedmap_ortho` directory.*

---

## 2. Full 5-Fold Cross-Validation Workflow

The entire experimental pipeline, from generating pseudo-labels to training all five models for cross-validation, can be run with the following commands.

### Step 2.1: Generate Pseudo-Label Maps for All Folds

This step runs the UnsemLabAG labeling pipeline on each of the five WeedMap orthomosaics. Each command uses a specific configuration file to handle the unique dimensions of each field.

**Note:** This is a computationally intensive process.

```bash
make generate CONFIG="config/config000.yaml" && \
make generate CONFIG="config/config001.yaml" && \
make generate CONFIG="config/config002.yaml" && \
make generate CONFIG="config/config003.yaml" && \
make generate CONFIG="config/config004.yaml"
```
The generated pseudo-label maps will be saved in the `results/` directory.

### Step 2.2: Extract Patches from All Maps

Once the full-field pseudo-label maps are generated, this command will slice them and the original images into smaller patches suitable for training the neural network.

```bash
make map_to_images CONFIG="config/config000.yaml" && \
make map_to_images CONFIG="config/config001.yaml" && \
make map_to_images CONFIG="config/config002.yaml" && \
make map_to_images CONFIG="config/config003.yaml" && \
make map_to_images CONFIG="config/config004.yaml"
```
*Note: This example assumes the `DATA_PATH` is handled internally by the script based on the config. If you need to pass it as a variable as in your example, please adjust the command accordingly.*

### Step 2.3: Train All Models for 5-Fold Cross-Validation

This is the final step. It will launch five separate training runs. In each run, one field is used for testing, while the other four are used for training, following the 5-fold cross-validation scheme.

```bash
make train CONFIG="config/config000.yaml" && \
make train CONFIG="config/config001.yaml" && \
make train CONFIG="config/config002.yaml" && \
make train CONFIG="config/config003.yaml" && \
make train CONFIG="config/config004.yaml"
```
The trained model checkpoints for each fold will be saved in their respective `experiments/` directory.

---

## How to Test a Trained Model

To evaluate a specific model checkpoint on its corresponding test set, use the `make test` command.

```bash
# Example: Test the best model from the fold where '000' was the test set
make test CHECKPOINT="./experiments/path/to/best_model_for_fold_0.ckpt" CONFIG="config/config000.yaml"
```
The evaluation results, including F1 scores, will be printed to the console.

---
## Citations

[1] **Unsupervised Semantic Label Generation in Agricultural Fields**
```bibtex
@article{roggiolani2025unsupervised,
  title={Unsupervised Semantic Label Generation in Agricultural Fields},
  author={Roggiolani, Gianmarco and R{\"u}ckin, Julius and Popovi{\'c}, Marija and Behley, Jens and Stachniss, Cyrill},
  journal={Frontiers in Robotics and AI},
  year={2025}
}
```

[2] **WeedMap: A Large-Scale Semantic Weed Mapping Framework...**
```bibtex
@article{sa2018weedmap,
  title={Weedmap: A large-scale semantic weed mapping framework using aerial multispectral imaging and deep neural network for precision farming},
  author={Sa, Inkyu and Popovi{\'c}, Marija and Khanna, Raghav and Chen, Zetao and Lottes, Philipp and Liebisch, Frank and Nieto, Juan and Stachniss, Cyrill and Walter, Achim and Siegwart, Roland},
  journal={Remote sensing},
  volume={10},
  number={9},
  pages={1423},
  year={2018},
  publisher={MDPI}
}
```