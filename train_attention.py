#!/usr/bin/env python3

import argparse
import chainer
import chainer.optimizers
import logging
import os
from lib import arguments
from lib import batch
from lib import model

def parse_args():
  p = argparse.ArgumentParser()
  arguments.add_file_args(p)
  arguments.add_batch_args(p)
  arguments.add_training_args(p)
  arguments.add_optimizer_args(p)
  arguments.add_gpu_args(p)
  arguments.add_attention_encdec_args(p)
  return p.parse_args()


def main():
  args = parse_args()

  if os.path.exists(args.output):
    raise RuntimeError('Directory or file %s already exists.' % args.output)
  os.makedirs(args.output)

  logging.basicConfig(
      filename=args.output + '/log',
      format='%(asctime)s\t%(name)s\t%(message)s',
      level=logging.DEBUG)
  logger = logging.getLogger(__name__)
  
  logger.info('Making new model:')
  logger.info('* src vocab size = %d', args.src_vocab_size)
  logger.info('* trg vocab size = %d', args.trg_vocab_size)
  logger.info('* embed size = %d', args.embed_size)
  logger.info('* hidden size = %d', args.hidden_size)
  logger.info('* attention size = %d', args.atten_size)
  mdl = model.AttentionEncoderDecoder(
      args.src_vocab_size,
      args.trg_vocab_size,
      args.embed_size,
      args.hidden_size,
      args.atten_size)

  if args.gpu >= 0:
    logger.info('Use GPU: ID = %d', args.gpu)
    chainer.cuda.get_device(args.gpu).use()
    mdl.to_gpu()
    model.use_gpu = True
  else:
    logger.info('Use CPU.')

  bos_id = 2
  eos_id = 3
  pad_id = 1
  logger.info('Making training batch generator:')
  logger.info('* source corpus = %s', args.train_src)
  logger.info('* target corpus = %s', args.train_trg)
  logger.info('* max sample length = %d', args.max_sample_length)
  logger.info('* max length ratio = %f', args.max_length_ratio)
  logger.info('* batch size = %d', args.train_batch_size)
  train_batches = batch.generate_train_batch(
      args.train_src, args.train_trg,
      pad_id, bos_id, eos_id,
      args.train_batch_size,
      args.max_sample_length,
      args.max_length_ratio)

  logger.info('Loading development corpus:')
  dev_batches = list(batch.generate_test_batch(
      args.dev_src, args.dev_trg,
      pad_id, bos_id, eos_id,
      args.test_batch_size))
  logger.info('Loading test corpus:')
  test_batches = list(batch.generate_test_batch(
      args.test_src, args.test_trg,
      pad_id, bos_id, eos_id,
      args.test_batch_size))

  logger.info('Making Adam optimizer:')
  logger.info('* learning rate = %f', args.learning_rate)
  logger.info('* gradient clipping = %f', args.gradient_clipping)
  logger.info('* weight decay = %f', args.weight_decay)
  opt = chainer.optimizers.Adam(alpha=args.learning_rate)
  opt.setup(mdl)
  opt.add_hook(chainer.optimizer.GradientClipping(args.gradient_clipping))
  opt.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

  trained_samples = 0
  logger.info('Start training:')
  logger.info('* total steps = %d', args.total_steps)
  logger.info('* evaluation interval = %d', args.eval_interval)
  logger.info('* max_generation_length = %d', args.max_generation_length)
  for step in range(args.total_steps):
    train_x_list, train_t_list = next(train_batches)
    mdl.zerograds()
    train_loss = mdl.forward_train(train_x_list, train_t_list)
    train_loss.backward()
    opt.update()
    trained_samples += len(train_x_list[0])
    step2 = step + 1

    if step2 % args.eval_interval == 0:
      logger.info('Step %d #trained samples = %d', step2, trained_samples)
      
      dev_accum_loss = 0.0
      dev_hyps = []
      for i, (dev_x_list, dev_t_list) in enumerate(dev_batches):
        dev_accum_loss += len(dev_x_list[0]) * float(
            mdl.forward_train(dev_x_list, dev_t_list).data)
        dev_z_list = mdl.forward_test(
            dev_x_list, bos_id, eos_id, args.max_generation_length)
        dev_hyps.extend(batch.batch_to_samples(dev_z_list, eos_id))
      logger.info('Step %d development loss = %.8e', step2, dev_accum_loss)
      with open(args.output + '/dev.hyp.%08d' % step2, 'w') as fp:
        for hyp in dev_hyps:
          print(' '.join(str(x) for x in hyp), file=fp)

      test_accum_loss = 0.0
      test_hyps = []
      for test_x_list, test_t_list in test_batches:
        test_accum_loss += len(test_x_list[0]) * float(
            mdl.forward_train(test_x_list, test_t_list).data)
        test_z_list = mdl.forward_test(
            test_x_list, bos_id, eos_id, args.max_generation_length)
        test_hyps.extend(batch.batch_to_samples(test_z_list, eos_id))
      logger.info('Step %d test loss = %.8e', step2, test_accum_loss)
      with open(args.output + '/test.hyp.%08d' % step2, 'w') as fp:
        for hyp in test_hyps:
          print(' '.join(str(x) for x in hyp), file=fp)
      

  logger.info('Finished.')


if __name__ == '__main__':
  main()
