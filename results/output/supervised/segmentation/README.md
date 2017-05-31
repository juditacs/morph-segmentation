# simple_seq2seq.out

This is the best parameter configuration so far.

## Command

   zcat $ULM_DIR/ulm/dat/webcorp/webcorp.100M.segmented.normalized2.gz | head -1000000 |  python run_experiment.py -l 20 --embedding-size 27 --save-test-output out2 --cell-size 256 --cell-type GRU --early-stopping-threshold 1e-4

### Explanation

| argument | explanation |
| ---- | ---- |
| l | maximum input length including segment boundaries (spaces) |
| embedding size | dimension of the embedding |
| cell type | GRU or LSTM |
| cell size | number of GRU/LSTM cells |

## Other parameters

* trained on 50000 samples, 80-10-10 train-valid-test split (so the training set is roughly 40000, random sampling was used)
* early stopping patience was 10
