import argparse
import collections
import sys
from ordered_set import OrderedSet
import os
import random
import re
import json
import logging
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from machinereading import settings
from machinereading.evaluation.util import load_input, read_quad, read_triple, str2bool


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OPT = None

TRAIN_PMID_FILE = 'distsup_pmid_split/train.txt'


def calc_f1(prec, rec):
  return 2 * (prec * rec) / (prec + rec)


def pr_plot(recall, precision, pr_auc, pr_image_file):
  # Create and save Precision-Recall Plot
  lw = 2
  plt.figure()
  plt.plot(recall, precision, color='darkorange',
           lw=lw, label='PR curve (area = %0.2f)' % pr_auc)
  plt.xlim([0.0, 1.05])
  plt.ylim([0.0, 1.05])
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('Precision-Recall')
  plt.legend(loc="upper right")
  plt.savefig(pr_image_file)


def pr_write(precision, recall, thresholds, pr_data_file):
  # Write precision recall threshold data to a file
  with open(pr_data_file, 'w') as fout:
    fout.write("Precision\tRecall\tTheshold\n")
    for prec, rec, thres in zip(precision, recall, thresholds):
      fout.write("{}\t{}\t{}\n".format(prec, rec, thres))


def quads_write(quads_dict, quads_file):
  # Write 'selected' quadruples along with their label and predictions to a file
  with open(quads_file, 'w') as fout:
    fout.write("sample_idx\tpmid\tdrug\tgene\tvariant\tprediction\tlabel\n")
    for key, value in quads_dict.items():
      pmid, drug, gene, variant = key
      prediction, label, sample_idx = value
      fout.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
          sample_idx, pmid, drug, gene, variant, prediction, label))


def metrics_at_pt5_prec(precision, recall, thresholds):
  # Print out metrics at 0.5 precision. Iterate from low to high threshold.
  print("Metrics at 0.5 precision")
  for prec, rec, thres in zip(precision, recall, thresholds):
    if np.around(prec, decimals=4) >= 0.5:
      F1 = calc_f1(prec, rec)
      print("Precision: {:.4f}\nRecall: {:.4f}\n"
            "Threshold: {:.4f}\nF1: {:.4f}\n".format(prec, rec, thres, F1))
      return prec, rec, thres, F1
  else:
    print("Precision never reaches 0.5")
    return None, None, None, None


def metrics_at_thresh(precision, recall, thresholds, threshold):
  # Print out standard metrics calculated at |threshold| threshold.
  print("Metrics at %f threshold" % threshold)
  for prec, rec, thres in zip(precision, recall, thresholds):
    if np.around(thres, decimals=4) >= threshold:
      F1 = calc_f1(prec, rec)
      print("Precision: {:.4f}\nRecall: {:.4f}\n"
            "Threshold: {:.4f}\nF1: {:.4f}\n".format(prec, rec, thres, F1))
      return prec, rec, thres, F1
  else:
    print("Could not find threshold = %f" % threshold)
    return None, None, None, None


def metrics_at_best_thresh(precision, recall, thresholds):
  # Print out standard metrics calculated at best threshold for F1.
  print("Metrics at best threshold")
  out_str = ''
  best_F1 = -1
  best_metrics = (None, None, None)
  for prec, rec, thres in zip(precision, recall, thresholds):
    if prec == 0.0 or rec == 0.0:
      F1 = 0.0
    else:
      F1 = calc_f1(prec, rec)
    if F1 > best_F1:
      best_F1 = F1
      best_metrics = (prec, rec, thres, F1)
      out_str = ("Precision: {:.4f}\nRecall: {:.4f}\n"
                 "Threshold: {:.4f}\nF1: {:.4f}\n".format(prec, rec, thres, F1))
  print(out_str)
  return best_metrics


def split_pmids(known_quads):
  with open(os.path.join(settings.DATA_DIR, 'jax/jax_dev_pmids.txt')) as f:
    dev_pmids = set(line.strip() for line in f)
  with open(os.path.join(settings.DATA_DIR, 'jax/jax_test_pmids.txt')) as f:
    test_pmids = set(line.strip() for line in f)
  return dev_pmids, test_pmids


def parse_args(args):
  parser = argparse.ArgumentParser(description="evaluate model from "
                                   "prediction files for quadruple instances")
  parser.add_argument('pred_file')
  parser.add_argument('--recall_correct', default='true')
  parser.add_argument('--skip_known_triples', default='true')
  parser.add_argument('--pr_data_file', default=None)
  parser.add_argument('--pr_image_file', default=None)
  parser.add_argument('--quads_file', default=None)
  parser.add_argument('--verbose', '-v', action='store_true')
  parser.add_argument('--test', action='store_true')
  parser.add_argument('--thresh', type=float)
  OPTS = parser.parse_args(args)
  OPTS.recall_correct = str2bool(OPTS.recall_correct)
  OPTS.skip_known_triples = str2bool(OPTS.skip_known_triples)
  return OPTS


def main(OPTS):
  random.seed(0)
  with open(os.path.join(settings.DATA_DIR, TRAIN_PMID_FILE)) as f:
    bad_pmids = set([line.strip() for line in f])
  pred_list = load_input(OPTS.pred_file, verbose=OPTS.verbose)

  known_quads = read_quad('181210_ckb_86genes_quads')
  dev_pmids, test_pmids = split_pmids(known_quads)
  cur_split_pmids = test_pmids if OPTS.test else dev_pmids
  known_quads = set(q for q in known_quads
                    if q[0] not in bad_pmids and q[0] in cur_split_pmids)

  known_triples = read_triple('181108_ckb_triples')

  quads_dict = {}
  for sample_idx, pmid, drug, gene, variant, prediction in pred_list:
    if pmid in bad_pmids:
      continue
    if pmid not in cur_split_pmids:
      continue
    quad = (pmid, drug, gene, variant)
    label = 1 if quad in known_quads else 0
    if OPTS.skip_known_triples and label == 0:
      triple = (drug, gene, variant)
      if triple in known_triples:
        if OPTS.verbose:
          print('Skipping sample_idx {}. known triple: '
                '{} {} {}'.format(sample_idx, drug, gene, variant))
        continue
    if quad not in quads_dict or (quad in quads_dict and
                                  quads_dict[quad][0] < prediction):
      quads_dict[quad] = (prediction, label, sample_idx)

  pred, label = zip(*[(pred_label[0], pred_label[1])
                      for quad, pred_label in quads_dict.items()])
  pos_count = sum(label)
  neg_count = len(label) - pos_count
  pos_found = pos_count / len(known_quads)
  precision, recall, thresholds = precision_recall_curve(label, pred)
  # Correcting the recall because it doesn't take into account positive
  # quadruples that we didn't even return
  if OPTS.recall_correct:
    recall = [x * pos_found for x in recall]

  print("=" * 50)
  pr_auc = auc(recall, precision)
  avg_prec = average_precision_score(label, pred)
  if OPTS.recall_correct:
    avg_prec *= pos_found
  print("PR AUC: {:.4f}\nAvgPrec: {:.4f}\nPositive Examples: {}\nNegative Examples: {}\n"
        "Positives Found(Max Recall): {:.1f}%\n".format(pr_auc, avg_prec, pos_count, neg_count,
                                            pos_found * 100))

  metrics_at_pt5_prec(precision, recall, thresholds)
  metrics_at_thresh(precision, recall, thresholds, 0.5)
  prec, rec, thres, F1 = metrics_at_best_thresh(precision, recall, thresholds)
  if OPTS.thresh:
    prec, rec, thres, F1 = metrics_at_thresh(
        precision, recall, thresholds, OPTS.thresh)

  if OPTS.pr_image_file:
    pr_plot(recall, precision, pr_auc, OPTS.pr_image_file)
  if OPTS.pr_data_file:
    pr_write(precision, recall, thresholds, OPTS.pr_data_file)
  if OPTS.quads_file:
    quads_write(quads_dict, OPTS.quads_file)
  print("=" * 50)
  return {'avg_prec': avg_prec, 'precision': prec, 'recall': rec,
          'threshold': thres, 'F1': F1, 'pr_auc': pr_auc, 'max recall': pos_found * 100}


if __name__ == '__main__':
  OPTS = parse_args(sys.argv[1:])
  main(OPTS)
