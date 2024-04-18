import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torchmetrics.functional.classification import binary_roc
from sklearn.metrics import f1_score, roc_curve, roc_auc_score
import torch

def plot_roc(labels, scores):
  # Assuming you have scores and labels as numpy arrays
  # scores: array containing predicted probabilities (shape: [n_samples])
  # labels: array containing true labels (shape: [n_samples])

  # Calculate ROC curve
  fpr, tpr, thresholds = binary_roc(scores, labels)
  roc_auc = auc(fpr, tpr)

  # Plot ROC curve
  plt.figure()
  plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic (ROC) Curve')
  plt.legend(loc="lower right")
  plt.show()


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

def f1_rejection_rate(y_true, y_pred, ood_s):
  rej_rates = []
  f1_scores = []

  for i in range(0, 101, 25):
    rej_rates.append(i/1000)
    r_labels, r_scores, _ = reject_per(y_true, y_pred, ood_s, i/1000)
    f1_scores.append(f1_score(r_labels, r_scores))

    if i%50 == 0 and i != 0:
      print(f"{i}% done")

  return rej_rates, f1_scores

def auroc_rejection_rate(y_true, scores, ood_s):
  rej_rates = []
  auroc_scores = []

  for i in range(0, 101, 25):
    rej_rates.append(i/1000)
    r_labels, r_scores = reject_per(y_true, scores, ood_s, i/1000)
    auroc_scores.append(roc_auc_score(r_labels, r_scores))

    if i%50 == 0 and i != 0:
      print(f"{i}% done")

  return rej_rates, auroc_scores

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


def detector_scores(detectors, datasets, device='cpu'):
  print(f"STAGE 3: Evaluating {len(detectors)} detectors on {len(datasets)} datasets.")

  results = {}

  with torch.no_grad():
      for detector_name, detector in detectors.items():
          print(f"> Evaluating {detector_name}")
          detector_results = {"scores": [],
                              "labels": [],
                              "logits": []
                              }
          for dataset_name, loader in datasets.items():
              print(f"--> {dataset_name}")

              for x, y in loader:
                  x = x.to(device)
                  y = y.to(device)
                  # Get logits
                  logits = detector.model(x).cpu()
                  # print(logits)

                  # print(logits)
                  # Get OOD scores
                  scores = detector.predict_features(logits).cpu()

                  #print(len(logits))

                  detector_results["scores"].append(scores)
                  detector_results["labels"].append(y)
                  detector_results["logits"].append(logits)
                  # print(detector_results["logits"])
                  # print(detector_results["labels"])
                  # print(detector_results["scores"])

              # detector_results["labels"] = np.array(detector_results["labels"]).ravel().tolist()
              # # print(l.shape)
              # detector_results["scores"] = np.array(detector_results["scores"]).ravel().tolist()
              # # print(ood_s.shape)
              # # print(np.array(detector_results["logits"]).reshape(-1, 2).shape)
              # detector_results["logits"] = np.array(detector_results["logits"]).reshape(-1, 2).tolist()
              detector_results["labels"] = torch.cat(detector_results["labels"]).tolist()
              # print(l.shape)
              detector_results["scores"] = torch.cat(detector_results["scores"]).tolist()
              # print(ood_s.shape)
              # print(np.array(detector_results["logits"]).reshape(-1, 2).shape)
              detector_results["logits"] = torch.cat(detector_results["logits"]).numpy()
          results[detector_name] = detector_results
  return results