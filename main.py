from load_data import Load_Data
from surprise import SVD


# Pick an arbitrary test subject
testSubject = 2

ld = Load_Data()

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

print("\nUser ", testSubject, " loved these movies:")
for ratings in loved:
    print(ld.getMovieName(ratings[0]))
print("\n...and didn't like these movies:")
for ratings in hated:
    print(ld.getMovieName(ratings[0]))

print("\nBuilding recommendation model...")
trainSet = data.build_full_trainset()

algo = SVD()
algo.fit(trainSet)

print("Computing recommendations...")
testSet = ld.BuildAntiTestSetForUser(testSubject, trainSet)
predictions = algo.test(testSet)

recommendations = []

print("\nWe recommend:")
for userID, movieID, actualRating, estimatedRating, _ in predictions:
    intMovieID = int(movieID)
    recommendations.append((intMovieID, estimatedRating))

recommendations.sort(key=lambda x: x[1], reverse=True)

for ratings in recommendations[:10]:
    print(ld.getMovieName(ratings[0]))
