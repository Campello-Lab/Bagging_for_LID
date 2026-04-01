<p align="center">
  <img src="bagging_figure/horizontalsimplebagging.png" width="1300" />
</p>


# Bagging for LID estimation

#### This repository has been developed for the publication with preprint published on [arXiv](https://arxiv.org/abs/2603.24384), as well as part of a master's thesis project viewable at [Master's Thesis](Papers/On_the_Use_of_Bagging_for_Local_Intrinsic_Dimensionality_Estimation__Master_s_Thesis.pdf).

- To recreate the results, plots, and figures present in the publication, first follow the [Installation](#installation) steps. Then do either of the following:

  - Follow the instructions at [Reproducibility](#reproducibility) to resample new datasets and perform the LID estimation experiments in the publication from scratch.
  - Follow the instructions at [Reproducibility](#reproducibility) to load ready-made light experiment objects from [light experiments](#Output/Complete_experiment_pkl_files/light_experiments), then recreate the plots and figures in the preferred style. Note that these are made only for plotting purposes and have no longer the necessary data attached to recreate the experiments from scratch.

- To use the Bagging_for_LID package for your own LID estimation experiments, or to examine experiment objects, first install the package via the [Installation](#installation) steps. Then follow the instructions at [Tutorial for the package](#tutorialforthepackage).

## Installation

#### Install the Bagging_for_LID package and its requirements.
 
To begin, clone the main repository to a selected folder

```bash
git clone https://github.com/anonymconference-star/Bagging_for_LID_Estimation.git
```

Navigate to selected folder

```bash
cd Bagging_for_LID
```

Install requirements. Python version $\geq 3.11$ required.

```bash
pip install -r requirements.txt
```

Install package

```bash
pip install -e .
```

## Reproducibility

#### Recreating results and figures from the publication [On the Use of Bagging for Local Intrinsic Dimensionality Estimation](https://arxiv.org/abs/2603.24384) 

- The [recreate_results_notebook](Reproducibility/recreate_results_notebook.ipynb) Jupyter notebook file contains detailed, step-by-step instructions on how to
recreate the results and figures present in the paper from scratch, or by loading already computed experiment objects.
- The [recreate_example_figures](Reproducibility/recreate_example_figures.ipynb) Jupyter notebook file can be used to recreate the plots in the Introduction section of the publication.
- The [recreate_results](Reproducibility/recreate_results.py) Python file can be used to recreate the same results without the use of a Jupyter notebook.

## Tutorial for the package

#### Using the Bagging_for_LID package to run your own LID estimation experiments

- The [single_experiment](Tutorials/single_experiment.ipynb) jupyter notebook file can be used to learn how to use the repository for examining or performing single LID estimation experiments, or multiple ones at once for a range of parameter combinations.





