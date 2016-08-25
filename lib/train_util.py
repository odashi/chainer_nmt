import chainer
import chainer.optimizers
import chainer.serializers
import logging
from . import batch
from . import model


BOS_ID = 1
EOS_ID = 2


def print_args(args):
  logger = logging.getLogger(__name__)

  logger.info('Arguments:')
  for key, val in sorted(vars(args).items(), key=lambda x: x[0]):
    logger.info('* %s = %s' % (key, val))


def init_simple_encdec_model(args):
  logger = logging.getLogger(__name__)

  logger.info('Making translation model:')
  logger.info('* class = SimpleEncoderDecoder')
  logger.info('* src vocab size = %d', args.src_vocab_size)
  logger.info('* trg vocab size = %d', args.trg_vocab_size)
  logger.info('* embed size = %d', args.embed_size)
  logger.info('* hidden size = %d', args.hidden_size)

  mdl = model.SimpleEncoderDecoder(
      args.src_vocab_size,
      args.trg_vocab_size,
      args.embed_size,
      args.hidden_size)

  return mdl


def init_atten_encdec_model(args):
  logger = logging.getLogger(__name__)

  logger.info('Making translation model:')
  logger.info('* class = AttentionEncoderDecoder')
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

  return mdl


def init_optimizer(args, mdl):
  logger = logging.getLogger(__name__)

  logger.info('Making Adam optimizer:')
  logger.info('* learning rate = %f', args.learning_rate)
  logger.info('* gradient clipping = %f', args.gradient_clipping)
  logger.info('* weight decay = %f', args.weight_decay)

  opt = chainer.optimizers.Adam(alpha=args.learning_rate)
  opt.setup(mdl)
  opt.add_hook(chainer.optimizer.GradientClipping(args.gradient_clipping))
  opt.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

  return opt


def load_params(prefix, mdl, opt):
  logger = logging.getLogger(__name__)

  logger.info('Loading model/optimizer parameters')
  chainer.serializers.load_npz(prefix + '.mdl', mdl)
  chainer.serializers.load_npz(prefix + '.opt', opt)


def save_params(prefix, mdl, opt):
  logger = logging.getLogger(__name__)

  logger.info('Saving model/optimizer parameters')
  chainer.serializers.save_npz(prefix + '.mdl', mdl)
  chainer.serializers.save_npz(prefix + '.opt', opt)


def prepare_gpu(args, mdl):
  logger = logging.getLogger(__name__)

  if args.gpu >= 0:
    logger.info('Use GPU: ID = %d', args.gpu)
    chainer.cuda.get_device(args.gpu).use()
    mdl.to_gpu()
    model.use_gpu = True
  else:
    logger.info('Use CPU.')


def prepare_data(args):
  logger = logging.getLogger(__name__)

  logger.info('Making training batch generator:')
  logger.info('* source corpus = %s', args.train_src)
  logger.info('* target corpus = %s', args.train_trg)
  logger.info('* max sample length = %d', args.max_sample_length)
  logger.info('* max length ratio = %f', args.max_length_ratio)
  logger.info('* batch size = %d', args.train_batch_size)

  train_batches = batch.generate_train_batch(
      args.train_src, args.train_trg,
      BOS_ID, EOS_ID,
      args.train_batch_size,
      args.max_sample_length,
      args.max_length_ratio)

  logger.info('Loading development corpus:')
  
  dev_batches = list(batch.generate_test_batch(
      args.dev_src, args.dev_trg,
      BOS_ID, EOS_ID,
      args.test_batch_size))

  logger.info('Loading test corpus:')
  
  test_batches = list(batch.generate_test_batch(
      args.test_src, args.test_trg,
      BOS_ID, EOS_ID,
      args.test_batch_size))

  return train_batches, dev_batches, test_batches


def train_step(mdl, opt, train_batch_gen):
  x_list, t_list = next(train_batch_gen)
  mdl.zerograds()
  loss = mdl.forward_train(x_list, t_list)
  loss.backward()
  opt.update()
  return len(x_list[0])


def test_model(mdl, batches, limit):
  accum_loss = 0.0
  hyps = []
  for x_list, t_list in batches:
    accum_loss += len(x_list[0]) * float(mdl.forward_train(x_list, t_list).data)
    z_list = mdl.forward_test(x_list, BOS_ID, EOS_ID, limit)
    hyps.extend(batch.batch_to_samples(z_list, EOS_ID))
  return accum_loss, hyps


def save_hyps(filename, hyps):
  with open(filename, 'w') as fp:
    for hyp in hyps:
      print(' '.join(str(x) for x in hyp), file=fp)
