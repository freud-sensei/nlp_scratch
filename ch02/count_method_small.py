import sys
import numpy as np
sys.path.append('..')
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, ppmi

text = "You say goodbye and I say hello."
corpus, word2id, id2word = preprocess(text)
vocab_size = len(id2word)
cmat = create_co_matrix(corpus, vocab_size)
pmat = ppmi(cmat)
U, S, V = np.linalg.svd(pmat)

# 결과값?
print(cmat[0])
print(pmat[0])
print(U[0])

# 밀집벡터의 차원을 2차원으로 줄인다면?
print(U[0, :2])

# 그래프로 그려보기
for word, id in word2id.items():
    plt.annotate(word, (U[id, 0], U[id, 1]))
plt.scatter(U[:, 0], U[:, 1], alpha=.5)
plt.show()