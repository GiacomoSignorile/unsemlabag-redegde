.PHONY: test

SHELL = /bin/sh

USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

GPUS ?= 0
MACHINE ?= default
CONFIG ?= ''
CHECKPOINT ?= 'weights.ckpt'
DATA_PATH := 
FOLDER := .
PATCH_SIZE ?= 224 # Default patch size, can be overridden
PATCHED_DATA_DIR := ./samples/RedEdge_Patches_$(PATCH_SIZE)

RUN_IN_CONTAINER = docker run -it --gpus all -e DISPLAY=$DISPLAY -v $(FOLDER):/unsemlabag unsemlab-ag

build:
	docker build . --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID) -t unsemlab-ag

download:
	$(RUN_IN_CONTAINER) bash -c "./download_assets.sh"

preprocess_patches:
	@echo "Preprocessing data into patches of size $(PATCH_SIZE)x$(PATCH_SIZE)..."
	@echo "Output will be in $(PATCHED_DATA_DIR)"
	$(RUN_IN_CONTAINER) python3 run_preprocessing.py # Assuming run_preprocessing.py uses PATCH_SIZE internally or you pass it as arg
	@echo "Patch preprocessing finished."

train:
	$(RUN_IN_CONTAINER) python3 train.py -c $(CONFIG)
	
generate:
	$(RUN_IN_CONTAINER) python3 main.py -c $(CONFIG)

test:
	$(RUN_IN_CONTAINER) python3 test.py -w $(CHECKPOINT) -c $(CONFIG)

map_to_images:
	$(RUN_IN_CONTAINER) python3 map_to_dataset.py -e $(DATA_PATH) -c $(CONFIG)

shell:
	$(RUN_IN_CONTAINER) "bash"

freeze_requirements:
	pip-compile requirements.in > requirements.txt
