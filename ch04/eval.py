import sys
sys.path.append('..')
from common.util import most_similar
import pickle
pkl_file = 'cbow_params.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word2id = params['word_to_id']
    id2word = params['id_to_word']

querys = ['you', 'year', 'car', 'toyota']
for q in querys:
    most_similar(q, word2id, id2word, word_vecs, top=5)
    
from common.util import analogy
analogy('king', 'man', 'queen', word2id, id2word, word_vecs)