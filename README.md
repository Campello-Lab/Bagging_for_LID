<p align="center">
  <img src="bagging_figure/horizontalsimplebagging.png" alt="Logo" width="1200" />
</p>


# Bagging for LID estimation

This repository has been developed for the publication: [On the Use of Bagging for Local Intrinsic Dimensionality Estimation](https://linktopaper)

## Installation

Clone the main repository to a selected folder

```bash
git clone https://github.com/pekristof/Bagging_for_LID.git
```

Navigate to selected folder

```bash
cd Bagging_for_LID
```

Install requirements

```bash
pip install -r requirements.txt
```

Install package

```bash
pip install -e .
```

## Recreating results

- The [recreate_results_notebook](recreate_results_notebook.ipynb) jupyter notebook file contains detailed, step-by-step instructions on how to
recreate the results and figures present in the paper from scratch, or by loading already computed experiment objects.
- The [recreate_results](recreate_results.py) python file can be used to recreate the same results without the use of jupyter notebook.
- The [recreate_example_figures](recreate_example_figures.ipynb) jupyter notebook file can be used to recreate the plots in the Introduction section of the publication.

## Data availability

- Zenodo link placeholder: The source files (.pkl) available at https://zenodo.org/ can be used together with our code to extract all the necessary information about the performed experiments, as well as to recreate the figures and the values in the tables. The files should be loaded and used by either [recreate_results_notebook](recreate_results_notebook.ipynb) or [recreate_results](recreate_results.py) and setting load = True.

- Don't forget to make this GitHub repo public once it is ready for submission.

- Don't forget to fix this README to the paths of the public github page





