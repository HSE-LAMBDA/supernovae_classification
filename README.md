# Classification of Type Ia Supernovae

This repository contains all the necessary data and scripts to reproduce the results of this article:

S. Dobryakov, K. Malanchev, D. Derkach and M. Hushchyn. Photometric Data-driven Classification of Type Ia Supernovae in the Open Supernova Catalog. [arXiv:2006.10489](https://arxiv.org/abs/2006.10489) [astro-ph.IM], 2020


## Instructions
- Install all packages listed in `requirements.txt`.
- To reproduce the paper results run `notebooks/article.ipynb` notebook.
- Then, run `notebooks/pred_for_types.ipynb` notebook to reproduce the plot with predictions for different object types.
- Experiments with PLASTiCC are described in `notebooks/plasticc_pipeline.ipynb` notebook.
- Experiments with Gaussian Processes for light curves augmentation are in `notebooks/gp_augmentation.ipynb` notebook.
- Preparation of OSC data is provided in `notebooks/osc_data_preparation.ipynb` notebook.
