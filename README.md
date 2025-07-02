<div align="center">
    <h1>Unsupervised Semantic Label Generation in Agricultural Fields</h1>
    <p><em>An adapted framework for evaluating unsupervised weed detection on the WeedMap dataset.</em></p>
    <br />
    <img src='pics/overview.png' width="800">
    <br/>
    <a href=https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/roggiolani2025frai.pdf>Original Paper</a>
    <span>  •  </span>
    <a href=https://github.com/PRBonn/unsemlabag/issues>Contact Us</a>
  <br />
  <br />
</div>

This repository contains the official code for the paper "Unsupervised Semantic Label Generation in Agricultural Fields". This version has been specifically adapted to run experiments on the **WeedMap RedEdge** dataset, allowing for a direct comparison with other state-of-the-art unsupervised methods like RoWeeder.

The framework simulates a drone's traversal over a large field orthomosaic, using computer vision heuristics to generate pseudo-ground truth labels, which are then used to train a semantic segmentation network.

---

## 1. Setup

### Environment

We recommend using Docker for a reproducible environment. You may need to [install the NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to use your GPU.

Build the Docker image:
```bash
makeOf build
```

### Download the WeedMap Dataset

This project is configured to work with the RedEdge subset of the WeedMap dataset.

``` course. The goal is to update the `README.md` for the **bash
# Create a directory for the data
mkdir -p dataset

# Download and unzip the dataset into the 'dataset' folder
wget http://robotics.ethz.ch/~asl-datasets/2018-weedUnsemLabAG** project to reflect your new work, which involved adapting it to run on the **WeedMap** datasetMap-dataset-release/Orthomosaic/RedEdge.zip -P dataset/
unzip dataset/RedEdge.zip -.

The new README needs to guide a user through the new workflow: downloading WeedMap, generating pseudo-labels for alld dataset/
```
Your directory should now look like this:
```
.
└── dataset/
    └── Red five fields, and then running the 5-fold cross-validation training. I will structure it clearly, keeping the original styleEdge/
        ├── 000/
        ├── 001/
        ...
```

---

## 2. Un but updating the commands and explanations for your specific use case.

---

### **Updated `README.md` for UnsemLabsupervised Label Generation Workflow

The core workflow consists of three main steps: generating a full-field pseudo-label map, patchAG on WeedMap**

````markdown
<div align="center">
    <h1>Unsupervised Semantic Label Generation inifying it into training samples, and finally, training the model. We provide `make` commands to automate the 5-fold cross-validation process Agricultural Fields</h1>
    <p><em>Adapting and Evaluating on the WeedMap Dataset</em></p>
    <br />
    <img src='pics/overview.png' width="700">
    <br/>
    <a href=https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf across all fields.

### Step 2.1: Generate Pseudo-Label Maps

This step runs the UnsemLabAG unsupervised pipeline on each of the five WeedMap orthomosaics. It uses the parameters defined in the `config/config<field_id>.yaml` files to generate a pseudo-GT image for each field.

**Note:** This is a computationally intensive process.

```bash
# This command will run the 'generate' target for each of the 5 config/roggiolani2025frai.pdf>Original Paper</a>
    <span>  •  </span>
    <a href=https://github.com/PRBonn/unsemlabag/issues>Contact Us</a>
    <br />
    <br />
</div>

This repository contains the official code for "Unsupervised Semantic Label Generation in Agricultural Fields files
make generate CONFIG="config/config000.yaml" && \
make generate CONFIG="config/config001.yaml" && \
make generate CONFIG="config/config002.yaml" && \
make generate CONFIG="config/config003.yaml" && \
make generate CONFIG="config/config004.yaml"
```
The resulting pseudo-label maps will be saved in the `results/` directory (e.g., `results/field_000_generated_label.png`).

### Step 2.2: Patch" \[[1](#citation)\]. This version has been adapted to run experiments on the **WeedMap** dataset \[[2](#citation)\], providing a benchmark and comparison for unsupervised labeling techniques.

The core method simulates a drone's flight over an orthomosaic, using computer vision heuristics to generate pseudo-labels, which are then used to train an evidential deep learning network for semantic segmentation.

---

## Setup

First, build the required Docker image.

```bash
make build
```
*You may need to [install the NVIDIA container toolkit](https://docs.nvidia.com/datacify Maps into a Dataset

Once the full-field pseudo-label maps are generated, this command will slice them into smaller image patches (`512x512` by default) suitable for training a neural network.

```bash
# This command runs the patch extraction for each generated map
make map_to_images DATA_PATH="results/000" CONFIG="config/config000.yaml" && \
make map_to_images DATA_PATH="results/001" CONFIG="config/config001.yaml" && \
make map_to_images DATA_PATHenter/cloud-native/container-toolkit/latest/install-guide.html) to enable GPU access within Docker.*

---

## Data Preparation for WeedMap

This workflow uses the Rheinbach subset of the WeedMap dataset.

### 1. Download WeedMap Dataset

Download the `RedEdge.zip` file containing the orthomosaics for all fields.

```bash
# This command will download and place the data in the correct directory.
make download="results/002" CONFIG="config/config002.yaml" && \
make map_to_images DATA_PATH="results/003" CONFIG="config/config003.yaml" && \
make map_to_images DATA_PATH="results/004" CONFIG="config/config004.yaml"
```
The patches will be saved in a structured format inside `results/generated/`.

---

## 3. Training and Evaluation

### Train the Model (5-Fold Cross-Validation)

_weedmap 
```
*Alternatively, you can manually download the [RedEdge data](http://robotics.ethz.ch/~asl-datasets/2018-weedMap-dataset-release/Orthomosaic/RedEdge.zip) and extract it to a `samples/weedmap_ortho` directory.*

### 2. (Optional but Recommended) Data Pre-processing

The original WeedMap orthomosaics have vertically oriented crop rows. The UnThis command will launch five separate training runs. Each run uses one field for testing and the other four for training, as defined in the corresponding `config<field_id>.yaml` file.

```bash
# This command runs thesemLabAG detector is configured by default to find horizontal rows. To ensure compatibility, we first rotate the orthomosaics. full training and validation for all 5 folds
make train CONFIG="config/config000.yaml" && \
make train CONFIG="config/config001.yaml" && \
make train CONFIG="config/config002.yaml" && \
make train CONFIG="config/config003.yaml" && \

```bash
# This will rotate all 5 fields and place them in samples/weedmap_rotated
make rotate_weedmap
```

---

## Full 5-Fold Cross-Validation Workflow

We have automated the entire 
make train CONFIG="config/config004.yaml"
```
Checkpoints for the best model from each fold will be saved in the `experiments/` directory.

### Test a Trained Model

To evaluate a specific model checkpoint5-fold cross-validation process using the `Makefile`. This involves generating pseudo-GT for each field, extracting patches on its corresponding test set, use the `make test` command.

```bash
# Example: Test the best, and training a model for each fold.

### 1. Generate Pseudo-Ground Truth for All Folds

This step runs the UnsemLabAG labeling pipeline on each of the five rotated WeedMap orthomosaics. It uses a model from the fold where '000' was the test set
make test CHECKPOINT=./experiments/path/to/best_model_for_fold_0.ckpt CONFIG="config/config000.yaml"
```
The separate configuration file (`config/config000.yaml`, etc.) for each field to handle its unique dimensions. evaluation results, including F1 scores, will be printed to the console.

---

## How to Customize

###

```bash
# This will take some time as it processes all five fields.
make generate_all_folds
```
The generated pseudo-label maps will be saved in the `results/` directory (e.g., `results/0 Using Your Own Data

If you want to train on a different dataset, you will need to:
1.  **Update00/field_000_generated_label.png`).

### 2. Extract Patches for the Makefile:** Change the `DATA_PATH` variable to point to your new data directory.
2.  **Write Training

Once the full-field pseudo-label maps are generated, this command will slice them and the original images into smaller a Dataloader:**
    a. Implement a new PyTorch `Dataset` class in the `datasets/` folder patches (e.g., 512x512) suitable for training the neural network.

```bash
.
    b. Import your new dataloader in `datasets/__init__.py`.
    c. In# This command extracts patches for all five fields.
make map_to_images_all_folds
```
The your YAML config, update `data.name` to match your new dataloader's name and `data.root_ patches will be saved in a structured format inside `results/generated_patches/`.

### 3. Train the Networkdir` to point to its location.
```` for All Folds

This is the final step. It will launch five separate training runs. In each run, one field is used for testing, while the other four are used for training and validation, following the 5-fold cross-validation scheme.

```bash
# This will train 5 separate models and save the checkpoints.
make train_all_folds
```
The trained model checkpoints for each fold will be saved in the `experiments/` directory.
