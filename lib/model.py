import chainer
import chainer.functions as F
import chainer.links as L
import numpy


use_gpu = False


def _array_module():
  return chainer.cuda.cupy if use_gpu else numpy


def _mkivar(array):
  m = _array_module()
  return chainer.Variable(m.array(array, dtype=m.int32))


def _zeros(shape):
  m = _array_module()
  return chainer.Variable(m.zeros(shape, dtype=m.float32))


class SimpleEncoderDecoder(chainer.Chain):
  def __init__(
      self,
      src_vocab_size,
      trg_vocab_size,
      embed_size,
      hidden_size):
    super(SimpleEncoderDecoder, self).__init__(
        # Encoder
        x_i = L.EmbedID(src_vocab_size, embed_size),
        i_p = L.Linear(embed_size, 4 * hidden_size, nobias=True),
        p_p = L.Linear(hidden_size, 4 * hidden_size),
        # Decoder initializer
        pc_qc = L.Linear(hidden_size, hidden_size),
        p_q = L.Linear(hidden_size, hidden_size),
        # Decoder
        y_j = L.EmbedID(trg_vocab_size, embed_size),
        j_q = L.Linear(embed_size, 4 * hidden_size, nobias=True),
        q_q = L.Linear(hidden_size, 4 * hidden_size),
        q_z = L.Linear(hidden_size, trg_vocab_size))
    self.src_vocab_size = src_vocab_size
    self.trg_vocab_size = trg_vocab_size
    self.embed_size = embed_size
    self.hidden_size = hidden_size

  def _encode(self, x_list):
    batch_size = len(x_list[0])
    pc = p = _zeros((batch_size, self.hidden_size))
    for x in reversed(x_list):
      i = self.x_i(_mkivar(x))
      pc, p = F.lstm(pc, self.i_p(i) + self.p_p(p))
    return pc, p

  def _initialize_decoder(self, pc, p):
    return F.tanh(self.pc_qc(pc)), F.tanh(self.p_q(p))

  def _decode_one_step(self, y, qc, q):
    j = self.y_j(_mkivar(y))
    qc, q = F.lstm(qc, self.j_q(j) + self.q_q(q))
    z = self.q_z(q)
    return z, qc, q

  def _loss(self, z, t):
    return F.softmax_cross_entropy(z, _mkivar(t))

  def forward_train(self, x_list, t_list):
    pc, p = self._encode(x_list)
    qc, q = self._initialize_decoder(pc, p)
    loss = _zeros(())
    for y, t in zip(t_list, t_list[1:]):
      z, qc, q = self._decode_one_step(y, qc, q)
      loss += self._loss(z, t)
    return loss

  def forward_test(self, x_list, bos_id, eos_id, limit):
    batch_size = len(x_list[0])
    pc, p = self._encode(x_list)
    qc, q = self._initialize_decoder(pc, p)
    z_list = []
    y = [bos_id for _ in range(batch_size)]
    while True:
      z, qc, q = self._decode_one_step(y, qc, q)
      z = [int(w) for w in z.data.argmax(1)]
      z_list.append(z)
      if all(w == eos_id for w in z):
        break
      elif len(z_list) >= limit: 
        z_list.append([eos_id for _ in range(batch_size)])
        break
      y = z
    return z_list


class AttentionEncoderDecoder(chainer.Chain):
  def __init__(
      self,
      src_vocab_size,
      trg_vocab_size,
      embed_size,
      hidden_size,
      atten_size):
    super(AttentionEncoderDecoder, self).__init__(
        # Encoder
        x_i = L.EmbedID(src_vocab_size, embed_size),
        i_f = L.Linear(embed_size, 4 * hidden_size, nobias=True),
        f_f = L.Linear(hidden_size, 4 * hidden_size),
        i_b = L.Linear(embed_size, 4 * hidden_size, nobias=True),
        b_b = L.Linear(hidden_size, 4 * hidden_size),
        # Attention
        fb_e = L.Linear(2 * hidden_size, atten_size, nobias=True),
        p_e = L.Linear(hidden_size, atten_size),
        e_a = L.Linear(atten_size, 1, nobias=True),
        # Decoder initializer
        fc_pc = L.Linear(hidden_size, hidden_size, nobias=True),
        bc_pc = L.Linear(hidden_size, hidden_size),
        f_p = L.Linear(hidden_size, hidden_size, nobias=True),
        b_p = L.Linear(hidden_size, hidden_size),
        # Decoder
        y_j = L.EmbedID(trg_vocab_size, embed_size),
        j_p = L.Linear(embed_size, 4 * hidden_size, nobias=True),
        q_p = L.Linear(2 * hidden_size, 4 * hidden_size, nobias=True),
        p_p = L.Linear(hidden_size, 4 * hidden_size),
        p_z = L.Linear(hidden_size, trg_vocab_size))
    self.src_vocab_size = src_vocab_size
    self.trg_vocab_size = trg_vocab_size
    self.embed_size = embed_size
    self.hidden_size = hidden_size
    self.atten_size = atten_size

  def _encode(self, x_list):
    batch_size = len(x_list[0])
    source_length = len(x_list)

    # Encoding
    fc = bc = f = b = _zeros((batch_size, self.hidden_size))
    i_list = [self.x_i(_mkivar(x)) for x in x_list]
    f_list = []
    b_list = []
    for i in i_list:
      fc, f = F.lstm(fc, self.i_f(i) + self.f_f(f))
      f_list.append(f)
    for i in reversed(i_list):
      bc, b = F.lstm(bc, self.i_b(i) + self.b_b(b))
      b_list.append(b)
    b_list.reverse()

    # Making concatenated matrix
    # {f,b}_mat: shape = [batch, srclen, hidden]
    f_mat = F.concat([F.expand_dims(f, 1) for f in f_list], 1)
    b_mat = F.concat([F.expand_dims(b, 1) for b in b_list], 1)
    # fb_mat: shape = [batch, srclen, 2 * hidden]
    fb_mat = F.concat([f_mat, b_mat], 2)
    # fbe_mat: shape = [batch * srclen, atten]
    fbe_mat = self.fb_e(
        F.reshape(fb_mat, [batch_size * source_length, 2 * self.hidden_size]))

    return fb_mat, fbe_mat, fc, bc, f_list[-1], b_list[0]

  def _context(self, p, fb_mat, fbe_mat):
    batch_size, source_length, _ = fb_mat.data.shape
    # {pe,e}_mat: shape = [batch * srclen, atten]
    pe_mat = F.reshape(
        F.broadcast_to(
            F.expand_dims(self.p_e(p), 1),
            [batch_size, source_length, self.atten_size]),
        [batch_size * source_length, self.atten_size])
    e_mat = F.tanh(fbe_mat + pe_mat)
    # a_mat: shape = [batch, srclen]
    a_mat = F.softmax(F.reshape(self.e_a(e_mat), [batch_size, source_length]))
    # q: shape = [batch, 2 * hidden]
    q = F.reshape(
        F.batch_matmul(a_mat, fb_mat, transa=True),
        [batch_size, 2 * self.hidden_size])

    return q

  def _initialize_decoder(self, fc, bc, f, b):
    return (
        F.tanh(self.fc_pc(fc) + self.bc_pc(bc)),
        F.tanh(self.f_p(f) + self.b_p(b)))

  def _decode_one_step(self, y, pc, p, fb_mat, fbe_mat):
    j = self.y_j(_mkivar(y))
    q = self._context(p, fb_mat, fbe_mat)
    pc, p = F.lstm(pc, self.j_p(j) + self.q_p(q) + self.p_p(p))
    z = self.p_z(p)
    return z, pc, p

  def _loss(self, z, t):
    return F.softmax_cross_entropy(z, _mkivar(t))

  def forward_train(self, x_list, t_list):
    fb_mat, fbe_mat, fc, bc, f, b = self._encode(x_list)
    pc, p = self._initialize_decoder(fc, bc, f, b)
    loss = _zeros(())
    for y, t in zip(t_list, t_list[1:]):
      z, pc, p = self._decode_one_step(y, pc, p, fb_mat, fbe_mat)
      loss += self._loss(z, t)
    return loss

  def forward_test(self, x_list, bos_id, eos_id, limit):
    batch_size = len(x_list[0])
    fb_mat, fbe_mat, fc, bc, f, b = self._encode(x_list)
    pc, p = self._initialize_decoder(fc, bc, f, b)
    z_list = []
    y = [bos_id for _ in range(batch_size)]
    while True:
      z, pc, p = self._decode_one_step(y, pc, p, fb_mat, fbe_mat)
      z = [int(w) for w in z.data.argmax(1)]
      z_list.append(z)
      if all(w == eos_id for w in z):
        break
      elif len(z_list) >= limit:
        z_list.append([eos_id for _ in range(batch_size)])
        break
      y = z
    return z_list

