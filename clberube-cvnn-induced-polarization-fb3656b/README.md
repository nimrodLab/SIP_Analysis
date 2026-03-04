# Complex-valued neural networks for spectral induced polarization applications
This repository hosts the necessary codes to reproduce experiments from the paper "Complex-valued neural networks for spectral induced polarization applications".

## Citation
Charles L Bérubé, Sébastien Gagnon, E Rachel Kenko, Jean-Luc Gagnon, Lahiru M A Nagasingha, Reza Ghanati, Frédérique Baron, Complex-valued neural networks for spectral induced polarization applications, Geophysical Journal International, 2025, ggaf348, https://doi.org/10.1093/gji/ggaf348 

## Necessary libraries (there may be more)
- scikit-learn https://scikit-learn.org/stable/index.html 
- pytorch https://pytorch.org/ 
- matplotlib https://matplotlib.org/ 
- numpy https://numpy.org/ 
- tqdm https://tqdm.github.io/ 

## Code structure
Each experiment is contained within its own directory:
- Experiment I: classip (classification of induced polarization spectra)
- Experiment II: parestim  (Cole-Cole parameter estimation)
- Experiment III: fapprox (mechanistic function approximation)

## How to run
In each experiment directory, the scripts should be run in order

1. Run `##_gen_data.py`
2. Run `train_mlp.py` 
3. Run the other scripts in order

## Other modules in each experiment directory
- `utilities.py` utility functions
- `plotlib.py` plotting helpers
- `models.py` neural networks 
- `seg.mplstyle` my matplotlib style sheet 
