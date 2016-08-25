#!/usr/bin/env python3

import argparse
import logging
import os
from lib import arguments
from lib import train_util


def main():
  p = argparse.ArgumentParser()
  arguments.add_file_args(p)
  arguments.add_batch_args(p)
  arguments.add_training_args(p)
  arguments.add_optimizer_args(p)
  arguments.add_gpu_args(p)
  arguments.add_simple_encdec_args(p)
  args = p.parse_args()

  if os.path.exists(args.output):
    raise RuntimeError('Directory or file %s already exists.' % args.output)
  os.makedirs(args.output)

  logging.basicConfig(
      filename=args.output + '/log',
      format='%(asctime)s\t%(name)s\t%(message)s',
      level=logging.DEBUG)
  logger = logging.getLogger(__name__)
  
  train_util.print_args(args)

  mdl = train_util.init_simple_encdec_model(args)
  opt = train_util.init_optimizer(args, mdl)
  train_util.prepare_gpu(args, mdl)
  train_batches, dev_batches, test_batches = train_util.prepare_data(args)

  trained_samples = 0
  logger.info('Start training:')
  for step in range(1, args.total_steps + 1):
    trained_samples += train_util.train_step(mdl, opt, train_batches)

    if step % args.eval_interval == 0:
      step_str = 'Step %d/%d' % (step, args.total_steps)
      logger.info('%s: #trained samples = %d', step_str, trained_samples)
      
      dev_accum_loss, dev_hyps = train_util.test_model(
          mdl, dev_batches, args.max_generation_length)
      logger.info('%s: dev loss = %.8e', step_str, dev_accum_loss)
      train_util.save_hyps(args.output + '/dev.hyp.%08d' % step, dev_hyps)

      test_accum_loss, test_hyps = train_util.test_model(
          mdl, test_batches, args.max_generation_length)
      logger.info('%s: test loss = %.8e', step_str, test_accum_loss)
      train_util.save_hyps(args.output + '/test.hyp.%08d' % step, test_hyps)

  logger.info('Finished.')


if __name__ == '__main__':
  main()
