### Source code for the Bayesion Machine Learning project on Stein Variational Gradient Descent

Analysis of the paper [A Non-Asymptotic Analysis for Stein Variational Gradient Descent](https://arxiv.org/pdf/2006.09797.pdf) for the Bayesian Machine Learning course at the MVA master at ENS Paris-Saclay.

One might want to read the [report](Report/main.pdf) first.

We also provide an implementation of the SVGD algorithm using PyTorch.

#### Usage
Install the required packages by running:
```bash
pip install numpy scipy seaborn tqdm matplotlib --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117
```
if you have a GPU, or
```bash
pip install numpy scipy seaborn tqdm matplotlib --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cpu
```
if you don't.

Then, run the code by running:
```bash
python main.py nb_iterations
```
it will run both experiments detailed in the report
and save all the figures in a `exp[1-2]` folder.