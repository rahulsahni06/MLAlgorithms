import math

import dataset.data_preprocessing as data_preprocessing
import dataset.dataset_utils as dataset_utils
import model_utils.model_utils as model_utils
from model_utils.naive_bayes import NaiveBayes
from model_utils.decision_tree import DecisionTree
from model_utils.knn import KNN, EUCLIDEAN, MANHATTAN
from model_utils.random_forest import RandomForest
import sys

if __name__ == "__main__":

    dataset = dataset_utils.LETTER_DATASET
    repeat_k_fold = 10
    k_fold = 5
    model_type = model_utils.RANDOM_FOREST
    accuracy_k_folds = []
    all_scores = []

    if model_type == model_utils.NAIVE_BAYES:
        model = NaiveBayes()
    elif model_type == model_utils.KNN:
        model = KNN(5, EUCLIDEAN)
    elif model_type == model_utils.RANDOM_FOREST:
        model = RandomForest(n_trees=20)
    else:
        model = DecisionTree()

    loaded_data, attribute_no, target_data_index, irrelevant_attributes, encode_categorical = dataset_utils\
        .get_dataset_settings(dataset, model_type)

    for i in range(repeat_k_fold):

        target_folds, attribute_folds = data_preprocessing \
            .split_k_folds(loaded_data, attribute_no, target_data_index, irrelevant_attributes, k_fold, encode_categorical)

        for j in range(k_fold):
            train_Y_data, train_X_data, test_Y_data, test_X_data = data_preprocessing \
                .get_train_test_data(target_folds, attribute_folds, j)
            if isinstance(model, RandomForest):
                model.set_n_sample_features(math.floor(math.sqrt(len(train_X_data))))

            model.train(train_X_data, train_Y_data)
            prediction = model.predict(test_X_data)
            acc = model.accuracy(prediction, test_Y_data)
            accuracy_k_folds.append(acc)
            sys.stdout.flush()
            print("\nTimes: {} Fold:{} Accuracy: {}".format(i+1, j+1, acc))

    avg_accuracy = sum(accuracy_k_folds)/len(accuracy_k_folds)
    print("\nAverage accuracy: {}".format(avg_accuracy))
    print("done")
