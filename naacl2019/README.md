# Document-Level N-ary Relation Extraction with Multiscale Representation Learning
This repository contains code for the following paper:

> Robin Jia, Cliff Wong, and Hoifung Poon.  
> **Document-Level N-ary Relation Extraction with Multiscale Representation Learning.**  
> _North American Association for Computational Linguistics (NAACL)_, 2019.  

### Note on terminology
While in the paper we refer to drugs, genes, and mutations,
in this codebase we refer to drugs, genes, and *variants* (i.e. D-G-V).

## Setup
1. Install `requirements.txt` using Python 3.6 and pip (tested on 3.6.7).
```
pip install -r requirements.txt 
```

## Main experiments
### Train and evaluate all The main 'triple' models in Table 3
Run the base sentence-level, paragraph-level, document-level models, along with their +Noisy-Or and +Noisy-Or+Gene Mutation filter variations. Saves the model and prediction outputs to an `out` directory. All the metrics reported in Table 3 are written to `out/log.txt`:
```
mkdir -p out
python -m machinereading.run --indir examples --gpu-id <gpuid>
```
Explanation:
* The first argument is the path directory that contains the preprocessed input data. This path is relative to the root data directory set in `machinereading\settings.py`
* The second argument should be an integer giving the ID of an available GPU on your machine (e.g., `0`). If you are using only CPU, replace `--gpu-id <gpuid>` with the parameter `--cpu-only`

### Train the individual 'triple' models
Alternatively, the steps below demonstrate how to run the steps for the individual sentence-level, paragraph-level, and document-level models, saving the output to an out directory:
```
./run.sh out/sentTriple_T20 sentence examples -T 20 --gpu-id <gpuid>  # sentence-level, D-G-V
./run.sh out/paraTriple_T20 paragraph examples -T 20 --gpu-id <gpuid>  # paragraph-level, D-G-V
./run.sh out/doc_T20 document examples -T 20 --gpu-id <gpuid>  # document-level, D-G-V
```
Explanation:
* The first argument is just a suggested name for an output directory, feel free to change it.
* The second argument is the text-level parameter (sentence, paragraph, or document). This breaks each document into text units of the specified size
* Additional flags are passed to `machinereading/models/backoffnet.py`. `-T 20` trains for 20 epochs, `--gpu-id <gpuid>` specifies the ID of an available GPU on your machine (e.g., `0`). If you are using only CPU, replace `--gpu-id <gpuid>` with the parameter `--cpu-only`.

### Development set evaluation
The output directories from the previous step will store JAX CKB predictions (for both development and test documents) from different checkpoints.
There will be versions without G-V filter (ending in `.tsv`) and with G-V filter (ending in `.tsv.pruned`).
You can evaluate any set of predictions on the dev set by running the evaluation script without any additional flags:
```
python3 -m machinereading.evaluation.eval out/paraTriple_T20/pred_dev_0018435.tsv

python3 -m machinereading.evaluation.eval out/doc_T20/pred_dev_0026896.tsv.pruned  # eval on G-V filtered dev set. Use average precision for model selection
python3 -m machinereading.evaluation.eval out/doc_T20/pred_dev_0026896.tsv  # eval on selected checkpoint dev set. Get threshold at best F1

```
This requires having the JAX data stored at `data/jax`. For each model, the checkpoint that has the highest average precision on the dev set after G-V filter is selected. The threshold is selected from evaluation on the dev set that gives the highest F1 (Metrics at best threshold). Save this threshold value and use it for test set evaluation to get the values reported in Table 3.

### Test set evaluation
Once you have chosen the checkpoints and threshold for each model, run the evaluation script on the test set with `--test` to get test-set numbers (with no noisy-or):
```
# Sentence-level
python3 -m machinereading.evaluation.eval out/sentTriple_T20/pred_test_0004652.tsv --test --thresh .5578 # No G-V filter
python3 -m machinereading.evaluation.eval out/sentTriple_T20/pred_test_0004652.tsv.pruned --test --thresh .5578 # With G-V filter

# Paragraph-level
python3 -m machinereading.evaluation.eval out/paraTriple_T20/pred_test_0018435.tsv --test --thresh .4595 # No G-V filter
python3 -m machinereading.evaluation.eval out/paraTriple_T20/pred_test_0018435.tsv.pruned --test --thresh .4595 # With G-V filter

# Document-level
python3 -m machinereading.evaluation.eval out/doc_T20/pred_test_0026896.tsv --test --thresh 0.6394 # No G-V filter
python3 -m machinereading.evaluation.eval out/doc_T20/pred_test_0026896.tsv.pruned --test --thresh 0.6394 # With G-V filter
```
See previous section on `Development set evaluation` for how we selected the checkpoint and threshold values.
In all cases, the example commands use the exact checkpoints and threshold we used in the paper. 
The numbers with no G-V filter should match the "base" numbers in Table 3 of the paper.
(We did not report the numbers with G-V filter but without noisy-or in the paper.)

### Noisy-or
To get the noisy-or results, first run `machinereading/models/ensemble.py` on the relevant `pred_test` file with the `log_noisy_or` option.
For example, to get results with noisy-or and G-V filter for the paragraph-level model, run: 
```
python3 -m machinereading.models.ensemble log_noisy_or out/paraTriple_T20/pred_dev_0018435.tsv.pruned > out/paraTriple_T20/pred_dev_paragraph_triple_logNoisyOr.tsv.pruned
python3 -m machinereading.models.ensemble log_noisy_or out/paraTriple_T20/pred_test_0018435.tsv > out/paraTriple_T20/pred_test_paragraph_triple_logNoisyOr.tsv.pruned
python3 -m machinereading.evaluation.eval out/paraTriple_T20/pred_dev_paragraph_triple_logNoisyOr.tsv.pruned
python3 -m machinereading.evaluation.eval out/paraTriple_T20/pred_test_paragraph_triple_logNoisyOr.tsv.pruned --test --thresh 0.3691
```

### MultiScale ensemble
To run the full MultiScale model, combine the predictions from these three smaller models, then evaluate:
```
# Ensemble with max, no G-V filter 
python3 -m machinereading.models.ensemble max out/sentTriple_T20/pred_dev_0004652.tsv out/paraTriple_T20/pred_dev_0018435.tsv out/doc_T20/pred_dev_0026896.tsv > out/pred_dev_multiscale_max.tsv
python3 -m machinereading.models.ensemble max out/sentTriple_T20/pred_dev_0004652.tsv out/paraTriple_T20/pred_dev_0018435.tsv out/doc_T20/pred_dev_0026896.tsv > out/pred_dev_multiscale_max.tsv
python3 -m machinereading.evaluation.eval out/pred_dev_multiscale_max.tsv # Get threshold at best F1
python3 -m machinereading.evaluation.eval out/pred_test_multiscale_max.tsv --test --thresh 0.7109

# Ensemble with noisy-or, no G-V filter 
python3 -m machinereading.models.ensemble log_noisy_or out/sentTriple_T20/pred_dev_0004652.tsv out/paraTriple_T20/pred_dev_0018435.tsv out/doc_T20/pred_dev_0026896.tsv > out/pred_dev_multiscale_logNoisyOr.tsv
python3 -m machinereading.models.ensemble log_noisy_or out/sentTriple_T20/pred_test_0004652.tsv out/paraTriple_T20/pred_test_0018435.tsv out/doc_T20/pred_test_0026896.tsv > out/pred_test_multiscale_logNoisyOr.tsv
python3 -m machinereading.evaluation.eval out/pred_dev_multiscale_logNoisyOr.tsv # Get threshold at best F1
python3 -m machinereading.evaluation.eval out/pred_test_multiscale_logNoisyOr.tsv --test --thresh 2.5892

# Ensemble with noisy-or, use G-V filter 
python3 -m machinereading.models.ensemble log_noisy_or out/sentTriple_T20/pred_dev_0004652.tsv.pruned out/paraTriple_T20/pred_dev_0018435.tsv.pruned out/doc_T20/pred_dev_0026896.tsv.pruned > out/pred_dev_multiscale_logNoisyOr.tsv.pruned
python3 -m machinereading.models.ensemble log_noisy_or out/sentTriple_T20/pred_test_0004652.tsv.pruned out/paraTriple_T20/pred_test_0018435.tsv.pruned out/doc_T20/pred_test_0026896.tsv.pruned > out/pred_test_multiscale_logNoisyOr.tsv.pruned
python3 -m machinereading.evaluation.eval out/pred_dev_multiscale_logNoisyOr.tsv.pruned # Get threshold at best F1
python3 -m machinereading.evaluation.eval out/pred_test_multiscale_logNoisyOr.tsv.pruned --test --thresh 1.1360
```

## Ablations
* To replace softmax with max, add the flag `--pool max` to the `run.sh` commands.
* To ensemble with only two models instead of all three just remove the corresponding predictions file from the call to `ensemble.py`.
```
# Minus Sentence-level
python3 -m machinereading.models.ensemble log_noisy_or out/paraTriple_T20/pred_dev_0018435.tsv.pruned out/doc_T20/pred_dev_0026896.tsv.pruned --out-file out/pred_dev_multiscale_minus_sentence_logNoisyOr.tsv.pruned
python3 -m machinereading.models.ensemble log_noisy_or out/paraTriple_T20/pred_test_0018435.tsv.pruned out/doc_T20/pred_test_0026896.tsv.pruned --out-file out/pred_test_multiscale_minus_sentence_logNoisyOr.tsv.pruned
python3 -m machinereading.evaluation.eval out/pred_dev_multiscale_minus_sentence_logNoisyOr.tsv.pruned # Save threshold at best F1
python3 -m machinereading.evaluation.eval out/pred_test_multiscale_minus_sentence_logNoisyOr.tsv.pruned --test --thresh 0.6162
```