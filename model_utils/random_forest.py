import random

from model_utils.model import Model
from model_utils.decision_tree import Node, DecisionTree


class RandomForest(Model):

    def __init__(self, n_trees=10, max_depth=50, min_instances=2, n_sample_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_instances = min_instances
        self.n_sample_features = n_sample_features
        self.forest = []

    def set_n_sample_features(self, n):
        self.n_sample_features = n

    def train(self, train_x_data, train_y_data):
        for i in range(self.n_trees):
            sample_x, sample_y = self.bagging_samples(train_x_data, train_y_data)
            decision_tree = DecisionTree(self.max_depth, self.min_instances, self.n_sample_features)
            decision_tree.train(sample_x, sample_y)
            self.forest.append(decision_tree)


    def predict(self, x_data):
        instance_count = len(x_data[0])
        prediction = []
        c = 1
        for instance_index in range(instance_count):
            instance = []
            for attribute_index in x_data:
                instance.append(x_data[attribute_index][instance_index])
            forest_prediction = {}
            for tree in self.forest:
                local_prediction = tree.traverse_tree(instance, tree.root)
                forest_prediction[local_prediction] = forest_prediction.get(local_prediction, 0) + 1

            prediction.append(sorted(forest_prediction, reverse=True, key=lambda x: forest_prediction[x])[0])
            Model.print_prediction_status(c, instance_count)
            c += 1
        return prediction


    def bagging_samples(self, train_x_data, train_y_data):
        n_instances = len(train_x_data[0])
        n_attributes = len(train_x_data)

        new_train_x_data = {}
        new_train_y_data = []

        for i in range(n_instances):
            sample_index = random.randint(0, n_instances-1)
            new_train_y_data.append(train_y_data[sample_index])

            for j in range(n_attributes):
                attributes_temp = new_train_x_data.get(j, None)
                if attributes_temp is None:
                    attributes_temp = []
                attributes_temp.append(train_x_data[j][sample_index])
                new_train_x_data[j] = attributes_temp

        return new_train_x_data, new_train_y_data

