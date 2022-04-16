from abc import abstractmethod
import sys

class Model:

    @staticmethod
    def accuracy(prediction_data, actual_data):
        total = 0
        correct = 0
        for prediction, actual in zip(prediction_data, actual_data):
            if prediction == actual:
                correct += 1
            total += 1
        return (correct/total) * 100

    @abstractmethod
    def predict(self, x_data):
        pass

    @abstractmethod
    def train(self, train_x_data, train_y_data):
        pass

    @staticmethod
    def print_prediction_status(current, total):
        percentage = (current/total) * 100
        # sys.stdout.write("\r{}/{} predicted".format(current, total))
        sys.stdout.write("\r{:0.2f}% Done".format(percentage))
        sys.stdout.flush()
        # print(f'Status: [{current}/{total}]')
