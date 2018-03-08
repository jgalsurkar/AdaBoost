# AdaBoost
Implement boosting for the Least Squares Classifier from scratch

The Least Squares Classifier performed least squares linear regression treating the ±1 labels as real-valued responses. The classifier is generally considered “weak”, and so boosting this classifier can be a good illustration of the method.

## Data

Information about the data used for this problem can be found [here](https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+)

## Technology Used
- Python 3

## Algorithm
![](./images/algorithm.png)

## Results

Error

![](./images/error_plot.png)

![](./images/upperbound_error.png)

Histogram of the total number of times each training data point was selected by the bootstrap method across all rounds (Sum the histograms of all B_t).

![](./images/training_points_hist.png)

Epsilon and alpha

![](./images/error_vs_t.png)

![](./images/alpha_v_t.png)



