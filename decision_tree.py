import numpy as np

class DecisionTree:
    def __init__(self, x_train, y_train, max_depth=20, min_split_data=5, max_attribute_threshold_candidate=20):
        self.max_depth = max_depth
        self.min_split_data = min_split_data
        self.x_train = x_train
        self.y_train = y_train
        self.nclass, = np.unique(self.y_train).shape
        self.max_attribute_threshold_candidate = max_attribute_threshold_candidate
        self.root = DecisionTreeNode(0, self.x_train, self.y_train, self.nclass,
                                     label=None, max_depth=self.max_depth,
                                     min_split_data=self.min_split_data,
                                     max_attribute_threshold_candidate=self.max_attribute_threshold_candidate)

    def train(self):
        self.root.split()

    def predict(self, x_test):
        predictions = []
        for x in x_test:
            predictions.append(self.root.predict(x))
        return np.array(predictions)

class DecisionTreeNode:
    def __init__(self, depth, x, y, nclass, label=None, max_depth=20, min_split_data=5, max_attribute_threshold_candidate=20):
        self.depth = depth
        self.x = x
        self.y = y
        self.ndata, self.nfeatures = self.x.shape
        self.nclass = nclass
        self.node_entropy = self.calculate_entropy(y, nclass)
        self.attribute_split_idx = None
        self.attribute_split_threshold = None
        self.label = label
        self.max_depth = max_depth
        self.min_split_data = min_split_data
        self.left_node = None
        self.right_node = None
        self.non_discriminative_attributes = []
        self.max_attribute_threshold_candidate = max_attribute_threshold_candidate

    def remove_non_discriminative_attributes(self):
        n = self.x.max() + 1
        temp = (np.arange(0, self.x.shape[1])[None, :])
        x_off = self.x + temp * n
        M = self.x.shape[1] * n
        b = (np.bincount(x_off.ravel(order='F'), minlength=M).reshape(-1, n) != 0).sum(1) != 1
        self.x = self.x[:, b]
        self.non_discriminative_attributes = np.where((b ^ 1).astype(bool))[0]

    def split_y_by_attribute_threshold(self, attribute_idx, threshold):
        mask = self.x[:, attribute_idx] > threshold
        attribute_data = [self.y[(mask ^ 1).astype(bool)], self.y[mask]]
        return attribute_data

    def calculate_attribute_split(self, attribute_idx):
        attribute_values = np.sort(np.unique(self.x[:, attribute_idx])).astype(float)
        threshold_candidates = None
        if (attribute_values.shape[0] - 1) <= self.max_attribute_threshold_candidate:
            threshold_candidates = np.array([(attribute_values[i] + attribute_values[i+1])/2 for i in range(attribute_values.size - 1)])
        else:
            idx_distance = (attribute_values.shape[0] - 1) // self.max_attribute_threshold_candidate
            threshold_candidates = np.array(
                [(attribute_values[i * idx_distance] + attribute_values[(i + 1) * idx_distance]) / 2 for i in range(self.max_attribute_threshold_candidate)]
            )
        information_gains = []
        for threshold in threshold_candidates:
            filtered_y = self.split_y_by_attribute_threshold(attribute_idx, threshold)
            entropies = []
            ndata_attributes = []
            for filtered_y_item in filtered_y:
                entropy = self.calculate_entropy(np.array(filtered_y_item), self.nclass)
                entropies.append(entropy)
                ndata_attributes.append(filtered_y_item.shape[0])
            information_gain = self.calculate_information_gain(np.array(ndata_attributes), np.array(entropies))
            information_gains.append(information_gain)

        information_gains = np.array(information_gains)
        optimal_idx = np.argmax(information_gains)
        return threshold_candidates[optimal_idx], information_gains[optimal_idx]

    def calculate_entropy(self, y, nclass):
        class_probs = np.zeros(nclass)
        ndata, _ = y.shape

        for i in range(nclass):
            class_probs[i] = float(len(y[y == i])) / ndata

        class_probs = class_probs[class_probs != 0]
        return -np.sum(class_probs * np.log2(class_probs))

    def calculate_information_gain(self, ndata_attributes, entropies):
        total_entropy_attribute = np.sum(np.multiply(ndata_attributes, entropies) / float(self.ndata))

        return self.node_entropy - total_entropy_attribute

    def split(self):
        if self.label != None:
            return

        # There is a case that there is only one
        # unique attributes value for all data on this node
        # I think this can be removed since the attribute doesn't
        # give any discriminative information
        self.remove_non_discriminative_attributes()

        # Terminal state
        if not self.x[0].size or \
                self.node_entropy == 0 or \
                self.depth >= self.max_depth or \
                self.y.size < self.min_split_data:
            counts = np.bincount(self.y[:, 0])
            self.label = np.argmax(counts)
            return

        attribute_thresholds = []
        attribute_information_gains = []
        for attribute_idx in range(self.x[0].size):
            threshold, information_gain = self.calculate_attribute_split(attribute_idx)
            attribute_thresholds.append(threshold)
            attribute_information_gains.append(information_gain)

        attribute_information_gains = np.array(attribute_information_gains)
        self.attribute_split_idx = np.argmax(attribute_information_gains)
        self.attribute_split_threshold = attribute_thresholds[self.attribute_split_idx]

        filter_data = self.x[:, self.attribute_split_idx] > self.attribute_split_threshold
        x_right = self.x[filter_data]
        x_left = self.x[~filter_data]
        y_right = self.y[filter_data]
        y_left = self.y[~filter_data]

        # Check if empty
        left_label = None
        right_label = None
        if not y_left.size:
            counts = np.bincount(self.y)
            left_label = np.argmax(counts)

        if not y_right.size:
            counts = np.bincount(self.y)
            right_label = np.argmax(counts)

        self.left_node = DecisionTreeNode(self.depth + 1, x_left, y_left, self.nclass,
                                          label=left_label, max_depth=self.max_depth,
                                          min_split_data=self.min_split_data)
        self.right_node = DecisionTreeNode(self.depth + 1, x_right, y_right, self.nclass,
                                           label=right_label, max_depth=self.max_depth,
                                           min_split_data=self.min_split_data)

        self.left_node.split()
        self.right_node.split()

    def predict(self, x_test):
        if self.label != None:
            return self.label

        # Delete non discriminative attributes
        for idx in reversed(self.non_discriminative_attributes):
            x_test = np.delete(x_test, idx, 0)

        if x_test[self.attribute_split_idx] > self.attribute_split_threshold:
            return self.right_node.predict(x_test)
        else:
            return self.left_node.predict(x_test)