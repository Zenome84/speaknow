
import numpy as np


class Evaluator:
    def __call__(self, filenames, groups, score_type, confidence_type):
        ratings = self.model(filenames, top_k=len(self.score_map), batch_size=8)

        if score_type == 'mean':
            score_func = self.__mean_score__
        elif score_type == 'argmax':
            score_func = self.__argmax_score__
        elif score_type == 'skewed_mean':
            score_func = lambda x: self.__skewed_mean_score__(x, 1.0)

        if confidence_type == 'std':
            confidence_func = self.__std_score__
        elif confidence_type == 'entropy':
            confidence_func = self.__entropy_score__

        scores = dict()
        confidences = dict()
        for n, rating in enumerate(ratings):
            scores.setdefault(groups[n], []).append(score_func(rating))
            confidences.setdefault(groups[n], []).append(confidence_func(rating))

        return scores, confidences

    def __mean_score__(self, rating):
        '''
        Gives a score that is the weighted mean value of each label
        '''
        return sum(
            self.score_map[rating['label']] * rating['score']
            for rating in rating
        )

    def __argmax_score__(self, rating):
        '''
        Gives a score that is the maximum rating likelihood
        '''
        return self.score_map[
            rating[
                np.argmax(
                    rating['score']
                    for rating in rating
                )
            ]['label']
        ]

    def __skewed_mean_score__(self, rating, beta):
        '''
        Similar to mean score, but penalizes by a high variability
        of scores, variability measured in standard deviation
        '''
        mean_score = self.__mean_score__(rating)
        std_score = np.sqrt(sum(
            (self.score_map[rating['label']] - mean_score)**2 * rating['score']
            for rating in rating
        ))
        return mean_score - beta * std_score

    def __std_score__(self, rating):
        '''
        Gives a score in [0, 1] that is proportional
        to the standard deviation of scores
        '''
        mean_score = self.__mean_score__(rating)
        std_score = np.sqrt(sum(
            (self.score_map[rating['label']] - mean_score)**2 * rating['score']
            for rating in rating
        ))
        score_distance = max(self.score_map.values()) - min(self.score_map.values())
        return 1 - 2*std_score/score_distance

    def __entropy_score__(self, rating):
        '''
        Gives a score in [0, 1] that is proportional
        to the entropy of the rating likelihoods
        '''
        probabilities = np.array([
            rating['score']
            for rating in rating
        ])
        effective_size = np.exp(-(np.log(probabilities + 1e-6)*probabilities).sum())
        effective_len = len(probabilities)
        return 1 - (effective_size-1)/effective_len

    @property
    def model(self):
        raise NotImplementedError("Child class of Evaluator must define a class or instance Model, set to '__model__'")
    
    @property
    def score_map(self):
        raise NotImplementedError("Child class of Evaluator must define a class or instance Score Map giving numerical values, set to '__score_map__'")
