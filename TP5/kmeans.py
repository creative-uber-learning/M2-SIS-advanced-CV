import numpy as np


class K_Means:
    def __init__(self, k=4, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for i,featureset in enumerate(data):
                distances = [np.linalg.norm(
                    featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(i)
            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                if(len(self.classifications[classification]) != 0):
                    self.centroids[classification] = np.average(data[self.classifications[classification]], axis=0)

            optimized = True
            
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    #print(np.sum((current_centroid-original_centroid) /original_centroid*100.0))
                    optimized = False

            if optimized:
                break
