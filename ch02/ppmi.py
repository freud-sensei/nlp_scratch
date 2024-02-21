import sys
import numpy as np
sys.path.append('..')
from common.util import preprocess, create_co_matrix, ppmi, cos_similarity

text = "You say goodbye and I say hello."
corpus, word2id, id2word = preprocess(text)
vocab_size = len(word2id)
cmat = create_co_matrix(corpus, len(word2id))
pmat = ppmi(cmat)

np.set_printoptions(precision=3) # 세 자릿수까지
print(cmat)
print(pmat)