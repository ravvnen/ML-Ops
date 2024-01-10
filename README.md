# 🎨 ML-Art

## Motivation

This project represents an application of machine learning(ML) in the domain of visual arts, where we are hoping to see how technology can be used to interpret and analyze artistic works. By training a model to recognize different art styles, we can help automate the process of sorting and labeling artworks, which traditionally has been a manual task. Moreover, this technology can potentially uncover patterns and influences between art styles that may not be immediately apparent to human observers, contributing to research in art history.

Furthermore, this classifier can serve as a foundation for recommendation systems in digital art platforms, providing users with suggestions of art styles they might enjoy based on their preferences.

Lastly, the cross-pollination of art and ML can lead to creative new forms of expression and digital art especially from Generative Artificial Intelligence(AI), where they are pushing the boundaries of both technology and artistic creation. This project is a step of where AI not only recognizing human creativity, but also augments and expands it.

## Data Analysis

Aiming at building a classifier that can accurately predict the art style of a given piece, we chose the [WikiArt Art Movement/Styles datatset](https://www.kaggle.com/datasets/sivarazadi/wikiart-art-movementsstyles/data), which contains 13 different styles with more then 42.5 thousands of images in 29.23 GB. The following plot shows the distribution of each style in the dataset, it is crucial for understanding the dataset's bias towards certain styles, here Romanticism is the most dominant and Western Medieval the least. Hence we would balanced the data for training our classifier to ensure that it performs well across all art styles, not just the most frequently occurring ones.

![distribution of styles](dist_styles.png)

Here we have also plotted a randomly selected image from each style, that hopefully can provide a better insight of each style and also make sure that our data is not flawed and contain instances from all styles.

![examples of styles](style_examples.png)


## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── dist_styles.png      <- The dataset visualization on distribution of each art styles.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── style_examples.png   <- The dataset visualization on examples from each art styles.
│
├── tests                <- Test files
│
├── ML-Art  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template), a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting started the course [Machine Learning Operations (MLOps) (02476)](https://kurser.dtu.dk/course/02476) offered by the [Technical University of Denmark](dtu.dk) in Jan 2024.
