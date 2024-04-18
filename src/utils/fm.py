import pickle
import json

def my_load(path, format='rb', module=pickle):
    with open(path, format) as f:
        object = module.load(f)
    return object


# TODO: Add documentation
def my_save(object, path, format='wb', module=pickle):
    with open(path, format) as f:
        module.dump(object, f)