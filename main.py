import time

import matplotlib.pyplot as plt

from metrics import Metrics
from decision_tree import DecisionTree
from digit_dataset import DigitDataset

def grid_search_decision_tree(file_path, max_depth_params, training_ratios):
    best_error = -1
    errors_train = {}
    errors_test = {}
    counter = 0.0
    dd = DigitDataset(file_path)

    for max_depth in max_depth_params:
        if errors_test.get(max_depth) == None:
            errors_test[max_depth] = {}
            errors_train[max_depth] = {}
        for training_ratio in training_ratios:
            if errors_test[max_depth].get(training_ratio) == None:
                errors_test[max_depth][training_ratio] = None
                errors_train[max_depth][training_ratio] = None

            x_train, y_train, x_test, y_test = dd.fit(training_ratio=training_ratio)
            dt = DecisionTree(x_train, y_train, max_depth=max_depth)
            dt.train()
            y_predict = dt.predict(x_train)
            error = 1.0 - Metrics.accuracy_score(y_train, y_predict)
            errors_train[max_depth][training_ratio] = error

            y_predict = dt.predict(x_test)
            error = 1.0 - Metrics.accuracy_score(y_test, y_predict)
            errors_test[max_depth][training_ratio] = error

            if error > best_error:
                best_error = error
                best_training_ratio = training_ratio
                best_max_depth = max_depth

        counter += 1.0
        print("Progress Grid Search Decision Tree: {}%".format(int(counter / len(max_depth_params) * 100)))

    return errors_train, errors_test, best_max_depth, best_training_ratio

def plot_2d(errors_train, errors_test, max_depth_params, training_ratios, filename):
    fig = plt.figure()
    for training_ratio in training_ratios:
        error_training_ratio = []
        for max_depth in max_depth_params:
            error_training_ratio.append(errors_train[max_depth][training_ratio])
        plt.plot(max_depth_params, error_training_ratio, label="Training error, training ratio {}".format(training_ratio))

    for training_ratio in training_ratios:
        error_training_ratio = []
        for max_depth in max_depth_params:
            error_training_ratio.append(errors_test[max_depth][training_ratio])
        plt.plot(max_depth_params, error_training_ratio, '--', label="Test error, training ratio {}".format(training_ratio))

    plt.legend(loc=1, bbox_to_anchor=(1.03, 1.1), fontsize='x-small')
    plt.xlabel("Max Depth")
    plt.ylabel("Error")
    plt.savefig(filename)

if __name__ == '__main__':
    # Read the Files
    file_path = "digits.mat"
    dd = DigitDataset(file_path)
    x_train, y_train, x_test, y_test = dd.fit()

    ###### HYPER PARAMETER SEARCHING ######
    max_depth_params = [5, 8, 11, 14, 17]
    training_ratios = [0.5, 0.6, 0.7, 0.8, 0.9]

    errors_train, errors_test, best_max_depth, best_training_ratio = grid_search_decision_tree(
        file_path,
        max_depth_params=max_depth_params,
        training_ratios=training_ratios
    )
    print("Best Decision Tree parameters max_depth: {}, min_split_data: {}".format(
        best_max_depth,
        best_training_ratio
    ))

    plot_2d(errors_train, errors_test, max_depth_params, [0.7], "4.2.png")
    plot_2d(errors_train, errors_test, max_depth_params, training_ratios, "4.3.png")

    print("Decision Tree Errors: ")
    print(errors_train)
    print(errors_test)

