from dataset.data_preprocessing import load_data
import model_utils.model_utils as model_utils

BREAST_DATASET = 1
CAR_DATASET = 2
LETTER_DATASET = 3
MUSROOM_DATASET = 4
ECOLI_DATASET = 5


def get_dataset_settings(dataset, model_type):
    if dataset == BREAST_DATASET:
        loaded_data = load_data('dataset/breast-cancer-wisconsin.data')
        attribute_no = 9
        target_data_index = 10
        irrelevant_attributes = [0]
        encode_categorical = False
    elif dataset == CAR_DATASET:
        loaded_data = load_data('dataset/car.data')
        attribute_no = 6
        target_data_index = 6
        irrelevant_attributes = []
        encode_categorical = False
        if model_type == model_utils.KNN:
            encode_categorical = True

    elif dataset == LETTER_DATASET:
        loaded_data = load_data('dataset/letter-recognition.data')
        attribute_no = 16
        target_data_index = 0
        irrelevant_attributes = []
        encode_categorical = False

    elif dataset == MUSROOM_DATASET:
        loaded_data = load_data('dataset/mushroom.data')
        attribute_no = 22
        target_data_index = 0
        irrelevant_attributes = [0]
        encode_categorical = False
        if model_type == model_utils.KNN:
            encode_categorical = True
    else:
        loaded_data = load_data('dataset/ecoli.data', separator=" ")
        attribute_no = 7
        target_data_index = 8
        irrelevant_attributes = [0]
        encode_categorical = False

    return loaded_data, attribute_no, target_data_index, irrelevant_attributes, encode_categorical
