from collections import Counter
from sklearn.base import BaseEstimator 

class KNN(BaseEstimator):
    def __init__(self, K):
       self.data = []
       self.K = K

    def fit(self, data, ids):
        self.data.extend(zip(data, ids))
        
    def predict(self, predData):
        result = []

        for query in predData:
            allDstQuer, closerIds = [], Counter() 

            for vec in self.data:
                dst = self.dist(vec[0], query)
                allDstQuer.append((dst, vec[1]))

            allDstQuer.sort()
            
            for i in range(self.K):
                vecId = allDstQuer[i][1]
                closerIds[vecId] += 1
            
            predId = closerIds.most_common(1)[0][0]
            result.append(predId)
        
        return result

    def dist(self, vec1, vec2):
        value, N = 0, min(len(vec1), len(vec2))

        for i in range(N):
            value += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i])

        return value
