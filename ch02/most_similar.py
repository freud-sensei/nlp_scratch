import sys
sys.path.append('..')
from common.util import preprocess, create_co_matrix, most_similar

text = "You say goodbye and I say hello."
corpus, word2id, id2word = preprocess(text)
cmat = create_co_matrix(corpus, len(word2id))
most_similar('you', word2id, id2word, cmat)