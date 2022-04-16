from abc import abstractmethod


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
