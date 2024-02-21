import sys
sys.path.append('..')
import numpy as np
from common.util import most_similar, create_co_matrix, ppmi
from dataset import ptb
from sklearn.utils.extmath import randomized_svd

window_size = 2
wordvec_size = 100 # 몇차원으로 압축?

corpus, word2id, id2word = ptb.load_data('train')
vocab_size = len(word2id)
print('동시발생 행렬 계산 중...')
cmat = create_co_matrix(corpus, vocab_size, window_size)
print('PPMI 계산 중...')
pmat = ppmi(cmat, verbose=True)
print('특이값 분해 중...')
U, S, V = randomized_svd(pmat, n_components=wordvec_size, n_iter=5)
word_vecs = U[:, :wordvec_size]

querys = ['you', 'year', 'car', 'toyota']
for q in querys:
    most_similar(q, word2id, id2word, word_vecs, top=5)