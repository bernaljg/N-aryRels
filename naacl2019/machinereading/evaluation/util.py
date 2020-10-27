"""Shared utilities."""
import collections
from collections import defaultdict
import re
import os
import io
import csv
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import settings

PMID_FILES = {
    '181108_ckb_86genes_pmids': 'pmid_lists/181108_ckb_86genes_pmids.txt'
}

DRUG_FILES = {
    '181109_ckb_all_drugs': 'drug_lists/181109_ckb_all_drugs.txt'
}

GENE_FILES = {
    'ckb_86_genes': 'gene_lists/ckb_86_genes.txt'
}

QUAD_FILES = {
    '181210_ckb_86genes_quads': 'jax/181210_ckb_86_genes_quad.tsv',
}

TRIPLE_FILES = {
    '181108_ckb_triples': 'jax/181108_ckb_triples.txt'
}


def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def read_quad(source):
  with open(os.path.join(settings.DATA_DIR, QUAD_FILES[source])) as fin:
    results = set((row[0], row[1], row[2], row[3])
                  for row in csv.reader(fin, delimiter='\t'))
  return results


def read_triple(source):
  with open(os.path.join(settings.DATA_DIR, TRIPLE_FILES[source])) as fin:
    results = set((row[0], row[1], row[2])
                  for row in csv.reader(fin, delimiter='\t'))
  return results


def read_drug(source):
  results = {}
  with open(os.path.join(settings.DATA_DIR, DRUG_FILES[source])) as fin:
    for drug_name, drug_syns in csv.reader(fin, delimiter='\t'):
      drug_name = drug_name.lower()
      for drug_syn in drug_syns.split('|'):
        results[drug_syn.lower()] = drug_name
  return results


def read_gene(source):
  results = set()
  with open(os.path.join(settings.DATA_DIR, GENE_FILES[source])) as fin:
    for gene_name in fin:
      results.add(gene_name.strip().lower())
  return results


def read_pmid(source):
  with open(os.path.join(settings.DATA_DIR, PMID_FILES[source])) as fin:
    results = set((row[0])
                  for row in csv.reader(fin, delimiter='\t'))
  return results


def load_input(pred_file, verbose):
  allowed_pmids = read_pmid('181108_ckb_86genes_pmids')
  allowed_genes = read_gene('ckb_86_genes')
  allowed_drugs = read_drug('181109_ckb_all_drugs')

  pred_list = []
  with open(pred_file) as f_pred:
    for idx, line in enumerate(f_pred):
      sample_idx, pmid, drug, gene, variant, pred = line.strip().split('\t')
      drug = drug.lower()
      gene = gene.lower()
      if pmid not in allowed_pmids:
        if verbose:
          logging.warning("Sample idx: {} PMID {} is not in PMID list. Will be "
                        "skipped".format(sample_idx, pmid))
        continue
      if gene not in allowed_genes or drug not in allowed_drugs:
        if verbose:
          logging.warning("Sample idx: {} Gene {} is not in gene list. Will be "
                        "skipped".format(sample_idx, gene))
        continue
      pred_list.append((sample_idx, pmid, drug.lower(), gene.lower(),
                        variant.lower(), float(pred)))
  return pred_list
