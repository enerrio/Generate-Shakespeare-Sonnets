# Generate-Shakespeare-Sonnets
Using a character-level language model to generate sonnets in the style of William Shakespeare. This project was heavily inspired by chapter 8 of Francois Chollet's book [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python). I wanted to create a model similar to the one described in the book but trained on Shakespeare's sonnets instead and with files to make generating new sonnets easy for users.

## File Descriptions
* **Generate-Sonnets.ipynb:** Jupyter notebook including data loading and visualization, model training, and explaining the approach used and and results.
* **shakespeare_sonnet_model.h5:** Keras model trained on sonnet data. This model is originally created, trained, and saved in the **Generate-Sonnets.ipynb** notebook.
* **generate_sonnet.py:** Python file that can be used to generate new sonnets easily and quickly. See **Usage** section below for more info.
* **sonnets.txt:** Text file containing all of William Shakespeare's sonnets where each sonnet is separated by a two newline characters.
* **nlp.yml:** Anaconda environment file.

## Requirements
* Python (3.x)
* Numpy (1.12.1 and up)
* Keras (2.1.4 and up)
* Tensorflow (1.5.0 and up)

## Usage
The file **generate_sonnet.py** loads the Keras model trained in **Generate-Sonnets.ipynb** and automatically chooses a random seed text and generates 600 characters (average sonnet length) that will print out to your console. To use it simply navigate to the directory where the python file is located and run the following command.

`python generate_sonnet.py`

A newly generated sonnet will then print out to your console. You may get a warning message from Tensorflow about CPU instructions but you can ignore that.

## Anaconda Environment
I also included a copy of my Anaconda environment that I used for this project. It contains all the necessary libraries and can easily be loaded in if you have anaconda/conda installed.

`conda env create -f nlp.yml`

To activate environment:

* Windows: `activate nlp`
* Mac/Linux: `source activate nlp`

After activating the environment you can run the code in the Jupyter notebook or the generate_sonnet python file. But if you already have the above packages installed this step isn't necessary.
