from model_utils.model import Model
import math
import multiprocessing as mp

EUCLIDEAN = "euclidean"
MANHATTAN = "manhattan"


def distance(instance1, instance2, dist_type=EUCLIDEAN):
    dist = 0
    for x1, x2 in zip(instance1, instance2):
        # print(x1, x2)
        if "." in x1:
            x1 = float(x1)
        else:
            x1 = int(x1)
        if "." in x2:
            x2 = float(x2)
        else:
            x2 = int(x2)

        if dist_type == EUCLIDEAN:
            dist += math.pow(abs(x1-x2), 2)
        elif dist_type == MANHATTAN:
            dist += abs(x1-x2)
        else:
            raise ValueError("Invalid distance type")

    if dist_type == "euclidean":
        return math.sqrt(dist)
    elif dist_type == MANHATTAN:
        return dist


class KNN(Model):

    def __init__(self, k, distance_type):
        self.train_x_data = {}
        self.train_y_data = {}
        self.calculated_distance = {}
        self.k = k
        self.distance_type = distance_type

    def train(self, train_x_data, train_y_data):

        instance_count = len(train_x_data[0])

        for instance_index in range(instance_count):
            instance = []
            for attribute_index in train_x_data:
                instance.append(train_x_data[attribute_index][instance_index])
            self.train_x_data[instance_index] = instance

        self.train_y_data = train_y_data

    def predict(self, x_data):

        instance_count = len(x_data[0])
        prediction = []

        c = 1
        for instance_index in range(instance_count):
            instance = []
            for attribute_index in x_data:
                instance.append(x_data[attribute_index][instance_index])

            for i in self.train_x_data:
                self.calculated_distance[i] = distance(instance, self.train_x_data[i], self.distance_type)

            nn_distance = sorted(self.calculated_distance, key=lambda x: self.calculated_distance[x])[:self.k]
            nn_class = {}
            for nn_index in nn_distance:
                nn = self.train_y_data[nn_index]
                nn_class[nn] = nn_class.get(nn, 0) + 1

            prediction.append(sorted(nn_class, key=lambda x: nn_class[x])[0])
            Model.print_prediction_status(c, instance_count)
            c += 1

        return prediction


