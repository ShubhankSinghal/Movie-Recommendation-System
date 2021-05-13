# https://surprise.readthedocs.io/en/stable/dataset.html

import load_data as LD
from surprise import SVD
from surprise import KNNBaseline
from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from metrics import Metrics

# Pick an arbitrary test subject
testSubject = 2

# create an object to use methods
ld = LD.Load_Data()


print("Loading movie ratings...")
data = ld.loadRatingDataset()

"""
userRatings = ld.getUserRatings(testSubject)

loved = []
hated = []
for ratings in userRatings:
    if (float(ratings[1]) > 4.0):
        loved.append(ratings)
    if (float(ratings[1]) < 3.0):
        hated.append(ratings)

print("\nUser: ", testSubject, " loved these movies:")
for ratings in loved:
    print(ld.getMovieName(ratings[0]))
print("\n...and didn't like these movies:")
for ratings in hated:
    print(ld.getMovieName(ratings[0]))


print("\nBuilding recommendation model...")
# build_full_trainset: Do not split the dataset into folds and just return a trainset as is,
# built from the whole dataset.
trainSet = data.build_full_trainset()

# Using SVD() model for recommendation
algo = SVD()
algo.fit(trainSet)

print("Computing recommendations...")
# fetching anti_dataset to recommend, excluding movies rated by testsubject
testSet = LD.BuildAntiTestSetForUser(testSubject, trainSet)
# making prediction
predictions = algo.test(testSet)

print("\nWe recommend:")
recommendations = []
for userID, movieID, actualRating, estimatedRating, _ in predictions:
    intMovieID = int(movieID)
    recommendations.append((intMovieID, estimatedRating))

# sorting in descending order based on estimated rating
recommendations.sort(key=lambda x: x[1], reverse=True)

# finally recommending top 10 estimated high rating movies for particular testSubject.
for ratings in recommendations[:10]:
    print(ld.getMovieName(ratings[0]))
"""

print("\nComputing movie popularity ranks so we can measure novelty later...")
rankings = ld.getPopularityRanks()

print("\nComputing item similarities so we can measure diversity later...")
fullTrainSet = data.build_full_trainset()
sim_options = {'name': 'pearson_baseline', 'user_based': False}
simsAlgo = KNNBaseline(sim_options=sim_options)
simsAlgo.fit(fullTrainSet)

print("\nBuilding recommendation model...")
trainSet, testSet = train_test_split(data, test_size=.25, random_state=1)

algo = SVD(random_state=10)
algo.fit(trainSet)

print("\nComputing recommendations...")
predictions = algo.test(testSet)

print("\nEvaluating accuracy of model...")
print("RMSE: ", Metrics.RMSE(predictions))
print("MAE: ", Metrics.MAE(predictions))

print("\nEvaluating top-10 recommendations...")

# Set aside one rating per user for testing
LOOCV = LeaveOneOut(n_splits=1, random_state=1)

for trainSet, testSet in LOOCV.split(data):
    print("Computing recommendations with leave-one-out...")

    # Train model without left-out ratings
    algo.fit(trainSet)

    # Predicts ratings for left-out ratings only
    print("Predict ratings for left-out set...")
    leftOutPredictions = algo.test(testSet)

    # Build predictions for all ratings not in the training set
    print("Predict all missing ratings...")
    bigTestSet = trainSet.build_anti_testset()
    allPredictions = algo.test(bigTestSet)

    # Compute top 10 recs for each user
    print("Compute top 10 recs per user...")
    topNPredicted = Metrics.GetTopN(allPredictions, n=10)

    # See how often we recommended a movie the user actually rated
    print("\nHit Rate: ", Metrics.HitRate(topNPredicted, leftOutPredictions))

    # Break down hit rate by rating value
    print("\nrHR (Hit Rate by Rating value): ")
    Metrics.RatingHitRate(topNPredicted, leftOutPredictions)

    # See how often we recommended a movie the user actually liked
    print("\ncHR (Cumulative Hit Rate, rating >= 4): ",
          Metrics.CumulativeHitRate(topNPredicted, leftOutPredictions, 4.0))

    # Compute ARHR
    print("\nARHR (Average Reciprocal Hit Rank): ",
          Metrics.AverageReciprocalHitRank(topNPredicted, leftOutPredictions))

print("\nComputing complete recommendations, no hold outs...")
algo.fit(fullTrainSet)
bigTestSet = fullTrainSet.build_anti_testset()
allPredictions = algo.test(bigTestSet)
topNPredicted = Metrics.GetTopN(allPredictions, n=10)

# Print user coverage with a minimum predicted rating of 4.0:
print("\nUser coverage: ", Metrics.UserCoverage(
    topNPredicted, fullTrainSet.n_users, ratingThreshold=4.0))

# Measure diversity of recommendations:
print("\nDiversity: ", Metrics.Diversity(topNPredicted, simsAlgo))

# Measure novelty (average popularity rank of recommendations):
print("\nNovelty (average popularity rank): ",
      Metrics.Novelty(topNPredicted, rankings))
