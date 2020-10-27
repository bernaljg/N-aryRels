import os
import re
import sys
import glob
import json
import argparse

from machinereading.models import backoffnet
from machinereading.models import prune_pred_gv_map
from machinereading.models import ensemble

from machinereading.evaluation import eval


# Logging
log_file = None  # global log file


def log(msg):
  print(msg, file=sys.stderr)
  if log_file:
    print(msg, file=log_file)


def train(in_dir, text_level, gpu_id, out_dir, cpu_only, epochs=20, mode='dgv'):
  params = ['--ds-train-dev-file', os.path.join(in_dir, 'ds_train_dev.txt'),
            '--jax-dev-test-file', os.path.join(in_dir, 'jax_dev_test.txt'),
            '--text-level', text_level, '-o', out_dir, '-T', str(epochs),
            '--try-all-checkpoints']
  if mode == 'dv':
    params.extend(['--pair-only', 'drug,variant'])
  elif mode == 'dg':
    params.extend(['--pair-only', 'drug,gene'])
  elif mode == 'dgv':
    pass
  else:
    raise "invalid mode for training"
  if gpu_id != None:
    params.extend(['--gpu-id', gpu_id])
  if cpu_only:
    params.append('--cpu-only')
  OPTS = backoffnet.parse_args(params)
  backoffnet.main(OPTS)


def prune_gv(data_dir, file_pattern):
  input_files = glob.glob(os.path.join(data_dir, file_pattern))
  if len(input_files) == 0:
    raise "No files found in input directory"
  for input_file in input_files:
    output_file = input_file + '.pruned'
    OPTS = prune_pred_gv_map.parse_args([input_file, output_file])
    prune_pred_gv_map.main(OPTS)


def eval_predictions(predictions_file, mode='dev', threshold=None):
  params = [predictions_file]
  if threshold:
    params.extend(["--thresh", str(threshold)])
  if mode == 'test':
    params.append('--test')
  OPTS = eval.parse_args(params)
  metrics = eval.main(OPTS)
  return metrics


def model_selection(data_dir, use_pruned_threshold=False):
  best_avg_prec = 0
  best_ckpt = None
  threshold = None
  dev_pruned_files = glob.glob(os.path.join(
      data_dir, "pred_dev_[0-9]*.tsv.pruned"))
  if len(dev_pruned_files) == 0:
    raise "No dev GV pruned files in dev-dir"
  for dev_pruned_file in dev_pruned_files:
    print(dev_pruned_file)
    dev_pruned_metrics = eval_predictions(dev_pruned_file)
    avg_prec = dev_pruned_metrics['avg_prec']
    if avg_prec > best_avg_prec:
      best_avg_prec = avg_prec
      m = re.search(r'pred_dev_(\d+).tsv', dev_pruned_file)
      best_ckpt = m[1]
      if use_pruned_threshold:
        dev_file = dev_pruned_file
      else:
        dev_file = os.path.join(data_dir, "pred_dev_{}.tsv".format(best_ckpt))
      print(dev_file)
      dev_metrics = eval_predictions(dev_file)
      threshold = dev_metrics['threshold']
  return best_ckpt, threshold


def gen_ensemble(mode, infiles, outfile, weights=None):
  params = ['--out-file', outfile, mode] + infiles
  OPTS = ensemble.parse_args(params)
  ensemble.main(OPTS)


def ensemble_metrics(mode, data_dir, dev_infiles, test_infiles, ensemble_filename_suffix):
  dev_ensemble_file = os.path.join(
      data_dir, "pred_dev_{}".format(ensemble_filename_suffix))
  test_ensemble_file = os.path.join(
      data_dir, "pred_test_{}".format(ensemble_filename_suffix))
  gen_ensemble(mode, dev_infiles, dev_ensemble_file)
  gen_ensemble(mode, test_infiles, test_ensemble_file)
  dev_metrics = eval_predictions(dev_ensemble_file)
  threshold = dev_metrics['threshold']
  metrics = eval_predictions(test_ensemble_file, 'test', threshold)
  return metrics, dev_ensemble_file, test_ensemble_file


def log_metrics(model_name, metrics):
  log('\n' + model_name)
  for metric, value in metrics.items():
    log('{} {:.4f}'.format(metric, float(value)))
  log('\n')


def run_triple_models(OPTS, best_models):
  """ This function reproduces the triple drug, gene, mutation 
  results in Table 3 
  """
  outdirs = ['sentTriple_T20', 'paraTriple_T20', 'doc_T20']
  outdirs = [os.path.join(OPTS.out_dir, x) for x in outdirs]
  
  levels = ['sentence', 'paragraph', 'document']
  multiscale_dev_infiles = []
  multiscale_test_infiles = []

  # Base Version results
  for level, outdir in zip(levels, outdirs):
    in_dir = os.path.join(OPTS.indir, level)
    # Train DGV model
    train(in_dir, level, OPTS.gpu_id, outdir, OPTS.cpu_only)
    # Generate Pruned GV prediction files for DGV model
    prune_gv(outdir, "pred_dev_*.tsv")
    prune_gv(outdir, "pred_test_*.tsv")
    # Model Selection
    best_ckpt, threshold = model_selection(outdir)
    metrics = eval_predictions(os.path.join(outdir, "pred_test_{}.tsv"
                                            .format(best_ckpt)), 'test', threshold)
    metrics['ckpt'] = best_ckpt
    best_models[level + ' Level'] = metrics
    multiscale_dev_infiles.append(os.path.join(
        outdir, "pred_dev_{}.tsv".format(best_ckpt)))
    multiscale_test_infiles.append(os.path.join(
        outdir, "pred_test_{}.tsv".format(best_ckpt)))
    log_metrics(level + ' Level', metrics)
  
  # Multiscale of Base Versions
  metrics, dev_ensemble_file, test_ensemble_file = ensemble_metrics(
      'max', OPTS.out_dir, multiscale_dev_infiles, multiscale_test_infiles, 'multiscale_max.tsv')
  best_models['Multiscale Base'] = metrics
  log_metrics('Multiscale Base', metrics)

  # + Noisy-Or results
  for level, outdir in zip(levels, outdirs):
    dev_infiles = [os.path.join(outdir, "pred_dev_{}.tsv".format(
        best_models[level + ' Level']['ckpt']))]
    test_infiles = [os.path.join(outdir, "pred_test_{}.tsv".format(
        best_models[level + ' Level']['ckpt']))]
    metrics, dev_ensemble_file, test_ensemble_file = ensemble_metrics(
        'log_noisy_or', outdir, dev_infiles, test_infiles, level + '_triple_logNoisyOr.tsv')
    best_models[level + ' Level + Noisy-Or'] = metrics
    log_metrics(level + ' Level + Noisy-Or', metrics)
  # Multiscale of + Noisy-Or
  metrics, dev_ensemble_file, test_ensemble_file = ensemble_metrics(
      'log_noisy_or', OPTS.out_dir, multiscale_dev_infiles, multiscale_test_infiles, 'multiscale_logNoisyOr.tsv')
  best_models['Multiscale + Noisy-Or'] = metrics
  log_metrics('Multiscale + Noisy-Or', metrics)

  # + Noisy-Or + Gene-mutation filter results
  for level, outdir in zip(levels, outdirs):
    dev_infiles = [os.path.join(outdir, "pred_dev_{}.tsv.pruned".format(
        best_models[level + ' Level']['ckpt']))]
    test_infiles = [os.path.join(outdir, "pred_test_{}.tsv.pruned".format(
        best_models[level + ' Level']['ckpt']))]
    metrics, dev_ensemble_file, test_ensemble_file = ensemble_metrics(
        'log_noisy_or', outdir, dev_infiles, test_infiles, level + '_triple_logNoisyOr.tsv.pruned')
    best_models[level + ' Level + Noisy-Or + Gene-mutation Filter'] = metrics
    log_metrics(level + ' Level + Noisy-Or + Gene-mutation Filter', metrics)

  multiscale_dev_infiles = [x+'.pruned' for x in multiscale_dev_infiles]
  multiscale_test_infiles = [x+'.pruned' for x in multiscale_test_infiles]
  # Multiscale of + Noisy-Or + Gene-mutation filter
  metrics, dev_ensemble_file, test_ensemble_file = ensemble_metrics(
      'log_noisy_or', OPTS.out_dir, multiscale_dev_infiles, multiscale_test_infiles, 'multiscale_logNoisyOr.tsv.pruned')
  best_models['Multiscale + Noisy-Or + Gene-mutation Filter'] = metrics
  log_metrics('Multiscale + Noisy-Or + Gene-mutation Filter', metrics)


def parse_args(args):
  parser = argparse.ArgumentParser()
  parser.add_argument('--indir', help='Training examples')
  parser.add_argument('--gpu-id', help='GPU ID to use for training')
  parser.add_argument('--cpu-only', action='store_true',
                      help='Run on CPU only')
  parser.add_argument('--out_dir', default='out/')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args(args)


def run(OPTS):
  best_models = {}
  run_triple_models(OPTS, best_models)


def main(OPTS):
  if not os.path.exists(OPTS.out_dir):
    os.makedirs(OPTS.out_dir)
  global log_file
  log_file = open(os.path.join(OPTS.out_dir, 'log.txt'), 'w')
  log(OPTS)
  try:
    run(OPTS)
  finally:
    if log_file:
      log_file.close()


if __name__ == '__main__':
  OPTS = parse_args(sys.argv[1:])
  main(OPTS)
