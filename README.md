# Anomaly-Detection-Simple
Simple Anomaly Detection system (Unsupervised Learning approach) using randomly generated data

## Description
Anomaly detection is a class of machine learning problems that takes in either a labeled (supervised learning) or unlabelled 
(unsupervised learning) dataset containing *m* examples and *n* features. This program generates an unlabelled toy dataset 
of random datapoints with a guassian distribution (**m = 50, n = 2**). Then a test dataset containing 5 examples is created 
and a threshold (epsilon) value is set. The probability of a test point being normal is calculated and if that probability 
is below the threshold then it is flagged and displayed in a plot.

## Dependencies
* numpy version 1.13.3 or greater
* matplotlib version 2.1.0 or greater
* Python 3.x

## Usage
Navigate to directory where *anomaly_detection.py* file is located using terminal of command line.
```python
python anomaly_detection.py
```
