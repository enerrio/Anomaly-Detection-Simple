#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def main():
    """
    Implement simple anomaly detection model.
    """

    # Create random train data
    data = np.random.normal(size=(50, 2))  # Mean of 0, std of 1
    mu = (1 / len(data)) * np.sum(data, axis=0)  # (2,)
    var = (1 / len(data)) * np.sum((data - mu) ** 2, axis=0)  # (2,)

    # Create 5 random test points
    test = np.random.normal(size=(5, 2))

    # Compute p(x) - i.e. probability that a given test data point is normal
    p_x = (1 / (np.sqrt(2 * np.pi) * np.sqrt(var))) * \
        np.exp(- ((test - mu) ** 2) / (2 * var))
    p_x = np.prod(p_x, axis=1)
    # Anomaly threshold
    epsilon = 0.03

    # Plot data
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], c="r", marker="x", label="Train data")
    plt.scatter(mu[0], mu[1], c="m", marker="o", label="Mean")
    plt.scatter(var[0], var[1], c="g", marker="v", label="Variance")
    anom_cnt = 0
    # If p(x) for any test point is below threshold, it is an anomaly
    for i in range(len(p_x)):
        if p_x[i] < epsilon:
            print("Anomaly Found!!")
            print(p_x[i])
            anom_cnt += 1
            plt.scatter(test[i, 0], test[i, 1], c="c", marker="X",
                        label="Anomaly_"+str(anom_cnt))
        else:
            plt.scatter(test[i, 0], test[i, 1], c="k", marker="X",
                        label="Test_"+str(i+1))
    plt.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    main()
