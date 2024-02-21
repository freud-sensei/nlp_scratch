import sys
sys.path.append('..')
from common.util import preprocess, create_co_matrix, cos_similarity

text = "You say goodbye and I say hello."
corpus, word2id, id2word = preprocess(text)
vocab_size = len(word2id)
c_matrix = create_co_matrix(corpus, vocab_size) # window size default는 1

c0 = c_matrix[word2id["you"]]
c1 = c_matrix[word2id["i"]]
print(cos_similarity(c0, c1))

