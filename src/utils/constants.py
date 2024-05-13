from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, RidgeClassifier, SGDOneClassSVM
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier


MODEL_CLASS_LIST = [
    LinearSVC,
    RandomForestClassifier
]

MODEL_NAME_LIST = [model_class.__name__ for model_class in MODEL_CLASS_LIST]
MODEL_NAME_TO_CLASS_DICT = {
    model_name: model_class
    for model_name, model_class in zip(MODEL_NAME_LIST, MODEL_CLASS_LIST)
}


_common_hparams = {
    'class_weight': {0: 1.0, 1: 10.0},
    'verbose': 1
}

HPARAMS_DICT = {
    'LinearSVC': {
        'C': 1,
        'class_weight': {0: 1.0, 1: 1.0}
    },
    'RandomForestClassifier': {
        'max_depth': 50,
        'class_weight': {0: 1.0, 1: 10.0}
    }
}

# for k in HPARAMS_DICT.keys():
#     HPARAMS_DICT[k].update(_common_hparams)




