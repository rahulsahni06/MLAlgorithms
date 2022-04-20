import math
import random
import sys
import time

from dataset.data_preprocessing import get_freq, most_common_class
from model_utils.model import Model


class Node:
    def __init__(self, attribute=None, threshold=None, target=None, left=None, right=None):
        self.attribute = attribute
        self.threshold = threshold
        self.target = target
        self.left = left
        self.right = right
        self.sample_weight = None

    def is_leaf(self):
        if self.target:
            return True
        return False


class DecisionTree(Model):

    def __init__(self, max_depth=50, min_instances=2, n_sample_features=None):
        self.max_depth = max_depth
        self.min_instances = min_instances
        self.root = None
        self.n_sample_features = n_sample_features

    def set_n_sample_features(self, n):
        self.n_sample_features = n

    @staticmethod
    def entropy(y, weight):
        if weight:
            freq = get_freq(weight)
            total = sum(weight)
        else:
            freq = get_freq(y)
            total = sum(freq.values())
        entropy = 0
        for element in freq:
            prob = freq[element]/total
            if prob > 0:
                entropy += prob * math.log2(prob)
        return -entropy

    def train(self, train_x_data, train_y_data, weight_data=None):
        # print("training")
        if self.n_sample_features is None:
            self.n_sample_features = len(train_x_data)
        self.root = self.grow_tree(train_x_data, train_y_data, weight_data=weight_data)
        # print("training done")

    def grow_tree(self, train_x_data, train_y_data, depth=0, weight_data=None):
        freq = get_freq(train_y_data)
        instance_size = len(train_x_data[0])

        if depth >= self.max_depth or len(freq) <= 1 or instance_size <= self.min_instances:
            target_class = most_common_class(train_y_data)
            return Node(target=target_class)

        attributes_indexes = self.get_random_attributes_index()
        best_attribute, best_threshold = self.get_best_criteria(train_x_data, train_y_data, attributes_indexes, weight_data)
        left_indexes, right_indexes = self.split_indexes(train_x_data, best_attribute, best_threshold)

        # print(best_attribute, depth, left_indexes, right_indexes)

        if len(left_indexes) == 0 or len(right_indexes) == 0:
            target_class = most_common_class(train_y_data)
            return Node(target=target_class)

        left_x_split = {}
        left_y_split = []
        right_x_split = {}
        right_y_split = []

        for i in left_indexes:
            for x in train_x_data:
                attribute_temp = left_x_split.get(x, None)
                if attribute_temp is None:
                    attribute_temp = []
                attribute_temp.append(train_x_data[x][i])
                left_x_split[x] = attribute_temp
            left_y_split.append(train_y_data[i])

        for i in right_indexes:
            for x in train_x_data:
                attribute_temp = right_x_split.get(x, None)
                if attribute_temp is None:
                    attribute_temp = []
                attribute_temp.append(train_x_data[x][i])
                right_x_split[x] = attribute_temp
            right_y_split.append(train_y_data[i])

        left = self.grow_tree(left_x_split, left_y_split, depth+1, weight_data)
        right = self.grow_tree(right_x_split, right_y_split, depth+1, weight_data)

        return Node(best_attribute, best_threshold, left=left, right=right)

    def get_random_attributes_index(self):
        indexes = []
        while len(indexes) < self.n_sample_features:
            n = random.randrange(0, self.n_sample_features)
            if n not in indexes:
                indexes.append(n)
        return indexes

    def predict(self, x_data):
        instance_count = len(x_data[0])
        prediction = []

        c = 1
        for instance_index in range(instance_count):
            instance = []
            for attribute_index in x_data:
                instance.append(x_data[attribute_index][instance_index])
            # sys.stdout.write("\r{:0.4f} Done".format(end-start))
            # sys.stdout.flush()
            prediction.append(self.traverse_tree(instance, self.root))

            Model.print_prediction_status(c, instance_count)
            c += 1

        return prediction

    def traverse_tree(self, instance, node):
        if node.is_leaf():
            return node.target
        if instance[node.attribute] <= node.threshold:
            return self.traverse_tree(instance, node.left)
        return self.traverse_tree(instance, node.right)

    def get_best_criteria(self, train_x_data, train_y_data, attributes_indexes, weight_data=None):
        best_gain = -1
        best_threshold = None
        best_attribute = None
        for i in attributes_indexes:
            threshold_freq = get_freq(train_x_data[i])
            for threshold in threshold_freq.keys():
                gain = self.information_gain(train_x_data, train_y_data, i, threshold, weight_data)
                if best_gain < gain:
                    best_gain = gain
                    best_attribute = i
                    best_threshold = threshold

        return best_attribute, best_threshold

    def information_gain(self, train_x_data, train_y_data, attribute_index, split_thresh, weight_data=None):

        parent_entropy = self.entropy(train_y_data, weight_data)

        left_indexes, right_indexes = self.split_indexes(train_x_data, attribute_index, split_thresh)

        if len(left_indexes) == 0 or len(right_indexes) == 0:
            return 0

        split_left_y_data = []
        split_right_y_data = []
        split_left_weight_data = []
        split_right_weight_data = []

        for i in left_indexes:
            split_left_y_data.append(train_y_data[i])
            if weight_data:
                split_left_weight_data.append(weight_data[i])

        for i in right_indexes:
            split_right_y_data.append(train_y_data[i])
            if weight_data:
                split_right_weight_data.append(weight_data[i])

        # compute the weighted avg. of the loss for the children
        n = len(train_y_data)
        n_l, n_r = len(left_indexes), len(right_indexes)
        e_l, e_r = self.entropy(split_left_y_data, split_left_weight_data), self.entropy(split_right_y_data, split_right_weight_data)
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig

    @staticmethod
    def split_indexes(train_x_data, attribute_index, threshold):
        left_indexes = []
        right_indexes = []

        for i, data in enumerate(train_x_data[attribute_index]):
            if data <= threshold:
                left_indexes.append(i)
            else:
                right_indexes.append(i)

        return left_indexes, right_indexes





