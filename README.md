# morph-segmentation-experiments

Experimenting with supervised morphological segmentation as a seq2seq problem

Currently two supervised models are supported: seq2seq and sequence tagger (LSTM) baseline.

# Seq2seq models

The seq2seq source code is located in the `morph_seg/seq2seq` directory.
It uses Tensorflow's `legacy_seq2seq`.

## Input data

Tre train script (`train.py`) expects the training input as either as its only positional argument or it reads it from standard input if no positional argument is provided.
Gzip files are supported.
The training data is expected to have one line per sample.
The input and the output sequences should be separated by TAB.

Example:

~~~
autót	autó t
ablakokat	ablak ok at
~~~

The inference script also reads from the standard input and expects one sample per line.

## Training your own model

~~~
cat training_data | python morph_seg/train.py --save-test-output test_output --save-model model_directory --cell-size 64 --result-file results.tsv
~~~

This will train a seq2seq model with the default arguments listed in the source file (train.py).

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

## Using your model for inference

`train.py` saves everything needed for inference to the directory specified by the `save-model` argument.
Inference can be run like this:

~~~
cat test_data | python morph_seg/inference.py --model-dir your_saved_model
~~~

Note that longer samples than the maximum length in the training data will be trimmed from their beginning.
