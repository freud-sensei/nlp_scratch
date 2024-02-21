import sys
sys.path.append('..')
import numpy as np
from common.functions import softmax
from ch06.rnnlm import Rnnlm
from ch06.better_rnnlm import BetterRnnlm

class RnnlmGen(Rnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        # skip_ids: 이 리스트에 속하는 단어는 샘플링되지 않음
        
        word_ids = [start_id]
        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape([1, 1])
            # 모델은 2차원 배열의 입력을 필요로 함
            score = self.predict(x)
            p = softmax(score.flatten())
            sampled = np.random.choice(len(p), size=1, p=p)
            
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))
        
        return word_ids