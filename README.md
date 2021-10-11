<div align = "center">
<h3>
Computer vision for anomaly detection in pictures
</h3>
<img width = "200" src = assets/dice.jpg alt="White dice">
</div>

<p align="center">
  <a href="#the-project">Project</a> •
  <a href="#data-source">Data source</a> •
  <a href="#our-solutions">Our solutions</a> •
  <a href="#how-to-use">How to use</a>
</p>

### The project

The aim of this (learning) project is to classify pictures of dice as **anomalous vs. normal**:
<div align = "center">
<img width = "200" src = assets/anomalous_synchro.gif alt="Anomalous dice">
<img width = "200" src = assets/normal_synchro.gif alt="Normal dice">
</div>
The dice on the left are anomalous while those on the right are normal.

### Data source

The images used to build this project have been provided by [Faktion](https://www.faktion.com) and are available [here](https://we.tl/t-Rfh9G5fseR).
Please grab these assets and extract them in the root directory.

### Our solutions

#### 1. OpenCV manipulations 
This first approach consists in comparing any input picture of a die to be classified as anomalous vs. normal with _templates_ of normal dice.
The idea being that if the input die is normal it should show a difference with the templates that is close to "zero". Whereas if the input die is anomalous, there should be a significant residual difference with the templates.

We found that the difference - in terms of pixels values - is a fair proxy of the dice class (normal or not) and allows to reach a 0.84 F1-score for the anomalous class on the original dataset.   

#### 2. Convolutional neural network (CNN) 
This second approach consists in ...

### How to use

You'll need [Python](https://www.python.org/) installed on your computer to clone and run this application.
From your command line:
```
# Clone this repository
$ git clone https://github.com/hakanErgin/faktion-usecase

# Go into the repository
$ cd faktion-usecase

# Install dependencies
$ pip install requirements.txt

# Run the main.py script
$ python run main.py
```


---
> GitHub
> [@hakanErgin](https://github.com/hakanErgin)
> [@lyesds](https://github.com/lyesds)
> [@nicesoul](https://github.com/nicesoul)
