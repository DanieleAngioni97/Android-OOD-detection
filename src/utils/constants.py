from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, RidgeClassifier, SGDOneClassSVM
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier


MODEL_CLASS_LIST = [
    # SVC,
    LinearSVC,
    LogisticRegression,
    Perceptron,
    SGDClassifier,
    RidgeClassifier,
    SGDOneClassSVM,
    GradientBoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier
]

MODEL_NAME_LIST = [model_class.__name__ for model_class in MODEL_CLASS_LIST]
MODEL_NAME_TO_CLASS_DICT = {
    model_name: model_class
    for model_name, model_class in zip(MODEL_NAME_LIST, MODEL_CLASS_LIST)
}

HPARAMS_DICT = {
    'LinearSVC': {
        'C': 1,
        # 'class_weight': 'balanced'
    },
    # 'SVC': {
    #     'C': 1,
    #     'kernel': 'linear'
    #     # 'class_weight': 'balanced'
    # },
    'LogisticRegression': {

    },
    'Perceptron': {
        # 'early_stopping': True,
        # 'validation_fraction': 0.2
    },
    'SGDClassifier': {
        'loss': 'hinge'

    },
    'RidgeClassifier': {

    },
    'SGDOneClassSVM': {

    },
    'GradientBoostingClassifier': {

    },
    'RandomForestClassifier': {

    },
    'AdaBoostClassifier': {

    }
}



FIT_HPARAMS_DICT = {
    'LinearSVC': {
        'C': 1,
        'class_weight': 'balanced'
    },
    'LogisticRegression': {

    },
    'Perceptron': {

    },
    'SGDClassifier': {

    },
    'RidgeClassifier': {

    },
    'SGDOneClassSVM': {

    },
    'GradientBoostingClassifier': {

    },
    'RandomForestClassifier': {

    },
    'AdaBoostClassifier': {

    }
}




