# Morphological segmentation

Experimenting with supervised morphological segmentation as a seq2seq problem.

Currently three supervised models are supported: seq2seq, sequence tagger using LSTM/GRU  and character-level CNN.

NOTE: I use the GitHub issue tracker for major bugs and enhancement proposals.
It is available in my [main repository](https://github.com/juditacs/morph-segmentation/issues).

## Setup

    pip install -r requirements.txt
    python setup.py install

## Input data

Tre training scripts (`train.py`) expect the training input as either as its only positional argument or it reads it from standard input if no positional argument is provided.
Gzip files are supported.
The training data is expected to have one line per sample.
The input and the output sequences should be separated by TAB.

Example:

~~~
autót	autó t
ablakokat	ablak ok at
~~~

The inference scripts (`inference.py`) also read from the standard input and expects one sample per line.

## seq2seq

The previous (working) seq2seq implementation used `legacy_seq2seq`.
The new implementation is still under heavy development.
Usage information will be added soon.


### Old (deprecated) seq2seq instructions

The code is available on te `seq2seq_old` branch.

The seq2seq source code is located in the `morph_seg/seq2seq` directory.
It uses Tensorflow's `legacy_seq2seq`.

#### Training your own model

~~~
cat training_data | python morph_seg/seq2seq/train.py --save-test-output test_output --save-model model_directory --cell-size 64 --result-file results.tsv
~~~

This will train a seq2seq model with the default arguments listed in `train.py`:

| argument | default | explanation |
| ----- | ----- | ------ |
| `save-test-output` | `None` | Save the model's output on the test set (randomly sampled) |
| `save-model` | `None` |  Save the model and other stuff needed for inference. This should be an exisiting directory. |
| `result-file` | `None` | Save the experiment's configuration and the result statistics. |
| `cell-type` | `LSTM` | Use LSTM or GRU cells. |
| `cell-size` | 16 | Number of LSTM/GRU cells to use. |
| `layers` | 1 | Number of layers. |
| `embedding-size` | 20 | Dimension of embedding. |
| `early-stopping-threshold` | 0.001 | Stop training when val loss does not change more than this threshold for N steps. |
| `early-stopping-patience` | 10 | Stop training if val loss does not change more than the threshold for N steps. |

Note that the first three arguments' default is `None`.
This means that unless specified, they do not write to file.
They are not linked though, any one can be left out.

#### Using your model for inference

`train.py` saves everything needed for inference to the directory specified by the `save-model` argument.
Inference can be run like this:

~~~
cat test_data | python morph_seg/seq2seq/inference.py --model-dir your_saved_model
~~~

Note that longer samples than the maximum length in the training data will be trimmed from their beginning.

## LSTM and CNN

This section needs updating.

The LSTM source code is located in the `morph_seg/sequence_tagger` directory.
It uses Keras's `LSTM`, `GRU` modules, and the usage is basically identical to the seq2seq model above.

### Training your own model

~~~
cat training_data | python morph_seg/sequence_tagger/train.py --config config.yaml --architecture RNN
~~~

See an example configuration at `config/sequence_tagger/toy.yaml`.

The majority of options is currently listed in the source code in `train.py`. Sorry :(

### Using your model for inference

~~~
cat test_data | python morph_seg/sequence_tagger/inference.py --model-dir your_saved_model
~~~
