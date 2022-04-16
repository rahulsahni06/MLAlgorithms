from model_utils.model import Model
import math


class NaiveBayes(Model):

    def __init__(self):
        self.class_freq = {}
        self.attribute_freq = {}
        self.log_prior_prob = {}
        self.no_of_attributes = 0

    def train(self, train_x_data, train_y_data):
        for y in train_y_data:
            self.class_freq[y] = self.class_freq.get(y, 0) + 1

        for attribute_index in train_x_data:
            if self.attribute_freq.get(attribute_index) is None:
                self.attribute_freq[attribute_index] = {}
            for x, y in zip(train_x_data[attribute_index], train_y_data):
                if self.attribute_freq.get(attribute_index).get(x) is None:
                    self.attribute_freq[attribute_index][x] = {}
                self.attribute_freq[attribute_index][x][y] = self.attribute_freq[attribute_index][x].get(y, 0) + 1

        total_class_count = sum(self.class_freq.values())
        for cls in self.class_freq:
            self.log_prior_prob[cls] = math.log(self.class_freq[cls]) - math.log(total_class_count)

        for attr in self.attribute_freq:
            self.no_of_attributes += len(self.attribute_freq[attr])

    def predict(self, x_data):
        prediction = []

        instance_count = len(x_data[0])

        c = 1
        for instance_index in range(instance_count):
            instance = []
            for attribute_index in x_data:
                instance.append(x_data[attribute_index][instance_index])

            prob = {}
            highest_prob = 0
            predicted_class = ""
            for cls in self.class_freq:
                prob[cls] = self.log_prior_prob[cls]

                for i in range(len(instance)):
                    try:
                        count = self.attribute_freq[i][instance[i]][cls]
                    except:
                        count = 0
                    prob[cls] += math.log(count + 1) - math.log(self.class_freq[cls] + self.no_of_attributes)

                highest_prob = prob[cls]
                predicted_class = cls

            for cls in prob:
                if highest_prob < prob[cls]:
                    highest_prob = prob[cls]
                    predicted_class = cls

            prediction.append(predicted_class)
            Model.print_prediction_status(c, instance_count)
            c += 1

        return prediction
