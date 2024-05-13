from utils import fm
import os
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np


def reject_per(l, s, ood_s, per):
  num_samples_to_reject = int(len(ood_s) * per)

  # Sort the OOD scores and get the indices of the highest scores
  sorted_indices = sorted(range(len(ood_s)), key=lambda i: ood_s[i], reverse=True)
  highest_indices = sorted_indices[:num_samples_to_reject]  # Select the indices of the two highest OOD scores

  # Reject the samples with the highest OOD scores

  rej_scores = [s[i] for i in range(len(s)) if i in highest_indices]
  rej_labels = [l[i] for i in range(len(l)) if i in highest_indices]

  remaining_scores = [s[i] for i in range(len(s)) if i not in highest_indices]
  remaining_labels = [l[i] for i in range(len(l)) if i not in highest_indices]

  # compute threshold

  if len(highest_indices) != 0:
    threshold = min([ood_s[i]for i in highest_indices])
  else:
    threshold = None

  # print("Remaining scores:", remaining_scores)
  # print("Remaining labels:", remaining_labels)

  return remaining_labels, remaining_scores, threshold

def reject_thr(logits, scores, ood_s, threshold):
  print(threshold)

  if threshold == None:
    return logits, scores, [], [], 0

  l_r = [logits[i] for i in range(len(ood_s)) if ood_s[i] < threshold]
  s_r = [scores[i] for i in range(len(ood_s)) if ood_s[i] < threshold]

  rej_labels = [logits[i] for i in range(len(ood_s)) if ood_s[i] >= threshold]
  rej_scores = [scores[i] for i in range(len(ood_s)) if ood_s[i] >= threshold]

  num_rej = len(logits) - len(l_r)

  return l_r, s_r, rej_labels, rej_scores, num_rej


path = "data/extended-features/"

results_list = fm.my_load(os.path.join(path, f'test_monthly_results.pkl'))
results_files = results_list

# Dictionary to store scores, labels, and logits for each detector
detector_data = {}

for i in range(len(results_files)):
  detector_data[str(i)] = {}
  # Iterate over each OOD detector
  for detector, info in results_files[i].items():
      scores = info['scores']
      labels = info['labels']
      logits = info['logits']

      # Save scores, labels, and logits for the detector
      detector_data[str(i)][detector] = {
          'scores': scores,
          'labels': labels,
          'logits': logits
      }

colors = [['#1f77b4', '#0e5b8c'], ['#ff7f0e', '#7e4c00'], ['#2ca02c', '#0f4614'], ['#d62728', '#670808']]

'''
  4 Plots for each test set, each containing 5 curves for the baseline with no rejection,
  then rejection based on the OOD scores (1% rejection rate)
'''

m_fpr = 1

# Initialize subplots
plt.rcParams.update({'font.size': 18})
num_of_ds = 4
y_dim = int(num_of_ds/2)
fig, axs = plt.subplots(2,y_dim, figsize=(20, 15*(num_of_ds/4)))


f1_rej_results = {}
thresholds = {}
rejected_samples = {"MaxSoftmax":[],
                    "EnergyBased":[],
                    "Entropy":[],
                    "OpenMax":[]}

for i, r_file_name in enumerate([results_files[0], results_files[45], results_files[46], results_files[47]]):
  r_file_name = str(i)
  f1_rej_results[r_file_name] = {}

  pred_l = [m[1] for m in detector_data[r_file_name]['OpenMax']["logits"]]
  l = detector_data[r_file_name]['OpenMax']["labels"]

  x, y, _ = roc_curve(l, pred_l)
  auroc_0 = roc_auc_score(l, pred_l, max_fpr=m_fpr)

  axs[i//y_dim][i%y_dim].plot(x, y, label=f'No Rej / {auroc_0:.3f}', alpha=0.7)
  axs[i//y_dim][i%y_dim].set_xscale('log')
  axs[i//y_dim][i%y_dim].set_xlabel('FP')
  axs[i//y_dim][i%y_dim].set_ylabel('TP')

  for detector, color in zip(detector_data[r_file_name].keys(), colors):
    print(detector)
    if i == 0:
      # Compute thresholds based on intial percentage 1% of the first dataset
      pred_l = [m[1] for m in detector_data[r_file_name][detector]["logits"]]
      l = detector_data[r_file_name][detector]["labels"]
      ood_s = detector_data[r_file_name][detector]["scores"]

      x, y, _ = roc_curve(l, pred_l)

      _, _, threshold = reject_per(l, pred_l, ood_s, .01)
      thresholds[detector] = threshold
      # print(detector_data[r_file_name][detector].keys())

    pred_l = [m[1] for m in detector_data[r_file_name][detector]["logits"]]
    l = detector_data[r_file_name][detector]["labels"]
    ood_s = detector_data[r_file_name][detector]["scores"]

    # print("th before being passed:", thresholds[detector])
    l_r, s_r, rej_l, rej_s, num_rej = reject_thr(l, pred_l, ood_s, thresholds[detector])

    rej_preds = [1 if x < 0 else 0 for x in rej_s]

    print(rej_preds)
    print(rej_l)
    misclassified_mw = sum(1 for pred, true in zip(rej_preds, rej_l) if pred != true)
    misclassified_gw = sum(0 for pred, true in zip(rej_preds, rej_l) if pred != true)

    print(
        f'# rejected goodware: {len([x for x in rej_l if x == 0])}',
        f'# rejected malware: {len([x for x in rej_l if x == 1])}',
        f'# rejected misclassfied goodware: {misclassified_gw}',
        f'# rejected misclassifed malware: {misclassified_mw}'
    )

    rejected_samples[detector].append([len([x for x in rej_l if x == 1]), len([x for x in rej_l if x == 0]), misclassified_mw])

    x, y, _ = roc_curve(l_r, s_r)
    auroc = roc_auc_score(l_r, s_r, max_fpr=m_fpr)

    # Detector / numeber of rejected samples / AUROC (max FPR 0.005)
    axs[i//y_dim][i%y_dim].plot(x, y, label=f'{detector} / {auroc:.3f} ({((auroc - auroc_0)/(auroc_0))*100:.3f}% increase)', alpha=0.7, color=color[0])
    axs[i//y_dim][i%y_dim].set_xscale('log')
    axs[i//y_dim][i%y_dim].set_xlabel('FPR')
    axs[i//y_dim][i%y_dim].set_ylabel('TPR')

  axs[i//y_dim][i%y_dim].set_title(f'Detectors in test set {r_file_name} - results\n (Detector / AUROC (max FPR .5%))')
  axs[i//y_dim][i%y_dim].legend()

# plt.tight_layout()
plt.savefig('test.png')
plt.close()

'''
  Plotting mean scores across different test datasets
'''

mean_scores_gw = []
mean_scores_mw = []
std_devs_gw = []
std_devs_mw = []

num_of_ds = 12*4

for i in range(num_of_ds):
  logits_gw = [m[1] for m in detector_data[str(i)]['OpenMax']["logits"] if m[1] < 0]
  logits_mw = [m[1] for m in detector_data[str(i)]['OpenMax']["logits"] if m[1] > 0]

  mean_scores_gw.append(np.mean(logits_gw))
  std_devs_gw.append(np.std(logits_gw))

  mean_scores_mw.append(np.mean(logits_mw))
  std_devs_mw.append(np.std(logits_mw))

# Define x-axis labels
x_labels = [str(i) for i in range(num_of_ds)]
print(len(mean_scores_gw))

plt.figure(figsize=(20,6))
plt.errorbar(x_labels, mean_scores_gw, yerr=std_devs_gw, fmt='o-', color='blue', label='Average Score for Goodware with Standard Deviation')
plt.errorbar(x_labels, mean_scores_mw, yerr=std_devs_mw, fmt='o-', color='red', label='Average Score for Malware with Standard Deviation')
plt.axhline(y=0, color='black', linestyle='dashed')

plt.xlabel('Test Sets')
plt.ylabel('Average Score')
plt.legend()

ax = plt.gca()
ax.get_xaxis().set_visible(False)

plt.savefig('test_mean_scores.png')