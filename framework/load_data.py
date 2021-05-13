import os
import csv
import sys
import re

from surprise import Dataset
from surprise import Reader

from collections import defaultdict
import numpy as np


class Load_Data:

    # These two dictionaries will be used to fetch id from movieName and vice versa.
    movieID_to_name = {}
    name_to_movieID = {}

    # defining relative path for the two dataset .CSV files
    ratingsPath = './dataset/ratings.csv'
    moviesPath = './dataset/movies.csv'

    def loadRatingDataset(self):
        """
          This function will load the two dataset files and return the rating dataframe
          and also populate the ID->movie and movie->ID dictionaries
        """

        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))

        # initialising default values
        ratingsDataset = 0
        self.movieID_to_name = {}
        self.name_to_movieID = {}

        # defining a Reader
        # line_format = columns names of rating.csv
        # comma seperated file
        # skip the header
        reader = Reader(line_format='user item rating timestamp',
                        sep=',', skip_lines=1)

        # loading data from rating.csv and storing it in ratingsDataset
        ratingsDataset = Dataset.load_from_file(
            self.ratingsPath, reader=reader)

        # Here, we will read from movies.csv and populate the two dictionaries
        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)  # Skip header line
            for row in movieReader:
                movieID = int(row[0])
                movieName = row[1]
                self.movieID_to_name[movieID] = movieName
                self.name_to_movieID[movieName] = movieID

        return ratingsDataset

    def getUserRatings(self, user):
        """
          This function will return the list of tuple (movieId, userRatings)
        """

        # initialising with default values
        userRatings = []
        # false until finds the user, then true if user matches, and after that, if user changes, it breaks the code.
        hitUser = False

        # appending movieId with rating to the userRatings list
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                userID = int(row[0])
                if (user == userID):
                    movieID = int(row[1])
                    rating = float(row[2])
                    userRatings.append((movieID, rating))
                    hitUser = True
                if (hitUser and (user != userID)):
                    break

        return userRatings

    def getPopularityRanks(self):
        """
            This function rank the movies based on their count of ratings
        """

        # initialised two dictionaries for our task
        ratings = defaultdict(int)
        rankings = defaultdict(int)

        # calculated total count of rating a movie get
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                movieID = int(row[1])
                ratings[movieID] += 1

        # providing rank to the movies, based on count
        rank = 1
        for movieID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[movieID] = rank
            rank += 1
        return rankings

    def getGenres(self):
        """
            This function returns the genre sparse list
        """
        genres = defaultdict(list)
        genreIDs = {}
        maxGenreID = 0

        # Here, fetch the positions for the genre for each movie
        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)  # Skip header line
            for row in movieReader:
                movieID = int(row[0])
                genreList = row[2].split('|')
                genreIDList = []
                for genre in genreList:
                    if genre in genreIDs:
                        genreID = genreIDs[genre]
                    else:
                        genreID = maxGenreID
                        genreIDs[genre] = genreID
                        maxGenreID += 1
                    genreIDList.append(genreID)
                genres[movieID] = genreIDList

        # Convert integer-encoded genre lists to bitfields that we can treat as vectors
        for (movieID, genreIDList) in genres.items():
            bitfield = [0] * maxGenreID
            for genreID in genreIDList:
                bitfield[genreID] = 1
            genres[movieID] = bitfield

        return genres

    def getYears(self):
        """
            This function returns the dictionary {movieId:Year}
        """
        # regular expression to extract year
        p = re.compile(r"(?:\((\d{4})\))?\s*$")
        years = defaultdict(int)
        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)
            for row in movieReader:
                movieID = int(row[0])
                title = row[1]
                m = p.search(title)
                year = m.group(1)
                if year:
                    years[movieID] = int(year)
        return years

    def getMovieName(self, movieID):
        """
            This function will return the movie name from movieID
        """
        if movieID in self.movieID_to_name:
            return self.movieID_to_name[movieID]
        else:
            return ""

    def getMovieID(self, movieName):
        """
            This function will return the movieId from movieName
        """
        if movieName in self.name_to_movieID:
            return self.name_to_movieID[movieName]
        else:
            return 0
