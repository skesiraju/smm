# Subspace Multinomial Model

* Learning document representations using subspace multinomial model. See [paper](https://www.google.cz/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwj8v4zkju7XAhUExxQKHceHA3EQFgguMAA&url=http%3A%2F%2Fwww.fit.vutbr.cz%2Fresearch%2Fgroups%2Fspeech%2Fpubli%2F2016%2Fkesiraju_interspeech2016_IS161634.pdf&usg=AOvVaw02jkh9Hzyo-Hpk36O2WX9l)
* This version of the code implements the same model, but with Adagrad optimization. This results in a slightly faster convergence with relatively lower memory requirements.

## Requirements
* `python3.6`
* `pytorch`, `numpy`, `scipy`, `scikit-learn`

## Data preparation

* `python TwentyNewsDataset.py`
* This will download the data from the web and converts it into `scipy.sparse` matrix.

## Training

* Input data: `scipy.sparse` matrix of shape `n_words x n_docs`
* `python run_smm_20news.py train -o exp/ -trn 100 -lw 1e-04 -rt l1 -lt 1e-4 -k 100`

* The trained model is saved as `exp/lw_1e-40_l1_1e-04_100/model_T100.pt`

### Positional parameters:
* `phase`: `train` or `extract`

### Hyper parameters:
* `-lw` : `l2` regularization const for i-vectors
* `-rt` : type of regularization for bases (`l1` or `l2`)
* `-lt` : regularization const for bases
* `-k`  : i-vector dimension
### Other options:
* `-o`  : path to output directory
* `-trn`: training iterations
* `--ovr`: over-write existing experiment directory

## Extracting i-vectors:

* `python run_smm_20news.py extract -m exp/lw_1e-04_l1_1e-04_100/model_T100.pt -xtr 30 --nth 2`

* The document i-vectors are saved in `exp/lw_1e-40_l1_1e-04_100/ivecs/`

### Other options:
* `-xtr`: extraction iterations.
* `--nth`: save every `n`-th i-vector while extraction.

## Classification using GLC

* `python train_and_clf.py exp/lw_1e-40_l1_1e-04_100/train_model_T100_e30.npy`
* Test data and labels are automatically read.

# On GPU

* prefix with `CUDA_VISIBLE_DEVICES=<device_id>` followed by `python run_smm_20news.py`