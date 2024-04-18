from utils import fm
import os


path = "../data/datasets_and_clf-10k/"

results_list = fm.my_load(os.path.join(path, f'test_results.pkl'))

print("")