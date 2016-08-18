import glob
import logging
import math
import os
import random
from collections import defaultdict


def _read_single_samples(filepath, bos_id, eos_id):
  with open(filepath) as fp:
    for line in fp:
      yield [bos_id] + [int(x) for x in line.split()] + [eos_id]


def _read_parallel_samples(src_filepath, trg_filepath, bos_id, eos_id):
  return zip(
      _read_single_samples(src_filepath, bos_id, eos_id),
      _read_single_samples(trg_filepath, bos_id, eos_id))


def _make_batch(samples, pad_id):
  batch_size = len(samples)
  max_src_length = max(len(sample[0]) for sample in samples)
  max_trg_length = max(len(sample[1]) for sample in samples)
  src_batch = [[pad_id] * batch_size for _ in range(max_src_length)]
  trg_batch = [[pad_id] * batch_size for _ in range(max_trg_length)]
  for i, (src_sample, trg_sample) in enumerate(samples):
    for j, w in enumerate(src_sample):
      src_batch[j][i] = w
    for j, w in enumerate(trg_sample):
      trg_batch[j][i] = w
  return src_batch, trg_batch


def _filter_samples(samples, max_sample_length, max_length_ratio):
  def _pred_max(x):
    return len(x[0]) <= max_sample_length and len(x[1]) <= max_sample_length

  def _pred_ratio(x):
    l0 = len(x[0])
    l1 = len(x[1])
    r = max(l0 / l1, l1 / l0)
    return r <= max_length_ratio
  
  samples = filter(_pred_max, samples)
  samples = filter(_pred_ratio, samples)
  return samples


def _arrange_samples(samples):
  buckets = defaultdict(lambda: [])
  for src, trg in samples:
    buckets[len(src)].append((src, trg))
  for key in sorted(buckets):
    samples_in_bucket = buckets[key]
    random.shuffle(samples_in_bucket)
    for sample in samples_in_bucket:
      yield sample


def _split_samples(samples, batch_size):
  batches = []
  for i in range(0, len(samples), batch_size):
    batches.append(samples[i : i + batch_size])
  return batches


def generate_train_batch(
    src_filepattern,
    trg_filepattern,
    bos_id,
    eos_id,
    batch_size,
    max_sample_length,
    max_length_ratio):
  logger = logging.getLogger(__name__)

  src_filepaths = sorted(glob.glob(src_filepattern))
  trg_filepaths = sorted(glob.glob(trg_filepattern))
  logger.info('#source filepaths = %d', len(src_filepaths))
  logger.info('#target filepaths = %d', len(trg_filepaths))
  assert src_filepaths
  assert trg_filepaths
  assert len(src_filepaths) == len(trg_filepaths)

  shard_ids = list(range(len(src_filepaths)))
  while True:
    random.shuffle(shard_ids)
    for shard_id in shard_ids:
      samples = _read_parallel_samples(
          src_filepaths[shard_id], trg_filepaths[shard_id], bos_id, eos_id)
      samples = _filter_samples(samples, max_sample_length, max_length_ratio)
      samples = _arrange_samples(samples)
      samples = list(samples)
      batches = _split_samples(samples, batch_size)
      random.shuffle(batches)
      logger.info('Loaded new shard:')
      logger.info('* source corpus = %s', src_filepaths[shard_id])
      logger.info('* target corpus = %s', trg_filepaths[shard_id])
      logger.info('* #filtered samples = %d', len(samples))
      logger.info('* #batches = %d', len(batches))
      for batch in batches:
        yield _make_batch(batch, eos_id)


def generate_test_batch(
    src_filepath,
    trg_filepath,
    bos_id,
    eos_id,
    batch_size):
  logger = logging.getLogger(__name__)
  samples = list(
      _read_parallel_samples(src_filepath, trg_filepath, bos_id, eos_id))
  batches = _split_samples(samples, batch_size)
  logger.info('Loaded files:')
  logger.info('* source corpus = %s', src_filepath)
  logger.info('* target corpus = %s', trg_filepath)
  logger.info('* #samples = %d', len(samples))
  logger.info('* #batches = %d', len(batches))
  for batch in batches:
    yield _make_batch(batch, eos_id)


def batch_to_samples(batch, eos_id):
  samples = [list(x) for x in zip(*batch)]
  samples = [x[ : x.index(eos_id)] for x in samples]
  return samples
