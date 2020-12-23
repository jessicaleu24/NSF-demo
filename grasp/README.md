
# README

# Install

1. Install requirements
	- `pip install -r requirements.txt`
2. Install models
	- `cd model/`
	- `python -m pip install -e detectron2`

3. Then, `exp_sim.py` should be ready to run

# Usage

- `exp_sim.py` provides an example for grasp planning
- Generally, pre-processed `depth` image and the camera `extrinsic` matrix should be given
- Grasp in the image plane can be transformed into 3D as shown in the example 