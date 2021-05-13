# https://surprise.readthedocs.io/en/stable/dataset.html

import load_data as LD
from surprise import SVD

# Pick an arbitrary test subject
testSubject = 2

# create an object to use methods
ld = LD.Load_Data()


print("Loading movie ratings...")
data = ld.loadRatingDataset()

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
