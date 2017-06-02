# morph-segmentation-experiments
Experimenting with supervised morphological segmentation as a seq2seq problem

# Experiment parameters

## Model

* embedding size
* cell type
* cell size
* multilayer rnn

### Training parameters

* batch size
* optimizer type
* optimizer parameters
* early stopping parameters
  * patience
  * threshold

## Data and feature extraction

* max samples
* uniq samples
* buckets
* use stop symbol

## Output

* data dimensions
* losses
  * by bucket size
* running time
* timestamp
* number of epochs run

## Environment

* machine
* GPU, Cuda version
* git commit hash
