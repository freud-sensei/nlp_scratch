import sys
sys.path.append('..')
import numpy as np
from common import config
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from cbow import CBOW
from common.util import create_contexts_target, to_cpu, to_gpu
from dataset import ptb

# hyperparams
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10

# dataset
corpus, word2id, id2word = ptb.load_data('train')
vocab_size = len(word2id)
contexts, target = create_contexts_target(corpus, window_size)

# model, optimizer
model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 학습 시작
trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

# 학습 데이터 저장
word_vecs = model.word_vecs
params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word2id'] = word2id
params['id2word'] = id2word
pkl_file = 'cbow.params.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)