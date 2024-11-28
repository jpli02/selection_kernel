# Selection kernel for self-attention 

## Requirements
```
pip install -r requirements.txt 
```
## Installation
```
pip install .
```

## Usage
### Forwarding function function 
```
out, c, LSE = selection_attention(q, k, v, causal, sm_scale)

```
out: standard weighted V

c: cumulate column-wise sum of attention score 

LSE: log sum of exponential for each row

### Running tests
1.  simple tests
```
python tests/tests.py --Z 10 --H 128 --N_CTX 1024 --HEAD_DIM 32
```

### Create sbatch jobs
1. `tests/job.slurm` creates sbatch files for running multiple experiments.

