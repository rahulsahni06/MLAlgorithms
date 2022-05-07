import math

from model_utils.model import Model
from model_utils.decision_tree import DecisionTree
from dataset.data_preprocessing import get_freq, most_common_class
import random

class AdaBoost(Model):

    def __init__(self, n_stumps=100):
        self.n_stumps = n_stumps
        self.weights = None
        self.alpha = []
        self.stumps = []
        self.target_classes_freq = {}

    def train(self, train_x_data, train_y_data):

        self.reset()
        self.target_classes_freq = get_freq(train_y_data)
        self.weights = self.assign_initial_weight(train_y_data)

        for i in range(self.n_stumps):

            if len(self.alpha) == 0:
                self.alpha = [0.0] * self.n_stumps

            stump = DecisionTree(max_depth=2)
            stump.train(train_x_data, train_y_data, self.weights)
            predicted_y_data = stump.predict(train_x_data)
            not_equal_matrix = self.not_equal_matrix(train_y_data, predicted_y_data)
            error = self.calculate_error(not_equal_matrix)
            self.alpha[i] = self.calculate_alpha(error)
            self.weights = self.update_weight(self.alpha[i], not_equal_matrix)
            self.stumps.append(stump)
        print()

    # def get_new_samples(self, train_x, train_y):
    #     max_range_weight = []
    #     new_train_x = {}
    #     new_train_y = []
    #     for idx, w in enumerate(self.weights):
    #         if idx == 0:
    #             max_range_weight[idx] = w
    #         max_range_weight[idx] = self.weights[idx-1] + w
    #
    #     for i in range(len(self.weights)):
    #         random_value = random.random()
    #         max = max_range_weight[i]
    #         min = max - self.weights[i]
    #         if min < random_value <= max:
    #             new_train_y[]


    def predict(self, x_data):
        instance_count = len(x_data[0])
        prediction = []

        c = 1
        for instance_index in range(instance_count):
            instance = {}
            for attribute_index in x_data:
                instance[attribute_index] = [x_data[attribute_index][instance_index]]

            stump_predictions = {}
            for i, stump in enumerate(self.stumps):
                pred = stump.predict(instance)[0]
                stump_predictions[pred] = stump_predictions.get(pred, 0) + self.alpha[i]
            prediction.append(most_common_class(stump_predictions))
            c += 1

        return prediction

    def calculate_error(self, not_equal_matrix):
        error = 0
        for i in range(len(not_equal_matrix)):
            error += self.weights[i] * not_equal_matrix[i]
        return error/sum(self.weights)

    def calculate_alpha(self, error):
        return math.log((1 - error)/error) + math.log(len(self.target_classes_freq) - 1)

    @staticmethod
    def assign_initial_weight(train_y_data):
        n = len(train_y_data)
        return [1/n] * n

    def update_weight(self, alpha, not_equal_matrix):
        new_weight = []
        for i in range(len(self.weights)):
            new_weight.append(self.weights[i] * math.exp(alpha * not_equal_matrix[i]))

        weight_total = sum(new_weight)
        for i in range(len(new_weight)):
            new_weight[i] = new_weight[i]/weight_total

        return new_weight


    @staticmethod
    def not_equal_matrix(train_y_data, predicted_y_data):
        I = []
        for x,y in zip(train_y_data, predicted_y_data):
            if x != y:
                I.append(1)
            else:
                I.append(0)
        return I


    def reset(self):
        self.weights = None
        self.alpha = []
        self.stumps = []
        self.target_classes_freq = {}
