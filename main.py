# https://surprise.readthedocs.io/en/stable/dataset.html

from framework.load_data import Load_Data
from surprise import SVD
from surprise import NormalPredictor
from framework.Evaluator import Evaluator
from framework.ContentKNNAlgorithm import ContentKNNAlgorithm

import random
import numpy as np


def LoadMovieLensData():
    ml = Load_Data()
    print("Loading movie ratings...")
    data = ml.loadRatingDataset()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)


np.random.seed(0)
random.seed(0)


# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadMovieLensData()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

# Throw in an SVD recommender
# SVDAlgorithm = SVD(random_state=10)
# evaluator.AddAlgorithm(SVDAlgorithm, "SVD")

contentKNN = ContentKNNAlgorithm()
evaluator.AddAlgorithm(contentKNN, "ContentKNN")

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ml)
