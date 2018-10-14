import numpy as np
import pandas as pd
from Abstracts import Abstracts
from sklearn.metrics.pairwise import cosine_similarity
import os.path

# -------------------------------------------------------------------------------------------------------
# PARAMETERS
# -------------------------------------------------------------------------------------------------------

# Authors to rank new abstracts against
authors = np.array(["emberson"])

# Input files containing author abstracts and previously checked abstracts 
input_dir       = "data/"
input_file_temp = input_dir + "author-%s-ids.dat"
past_sub_file   = input_dir + "submissions.dat" 

# arXiv category and maximum number of submissions to rank 
category = "astro-ph.CO"
nsub_max = 100

# Maximum number of submissions to print scores
nprint_max = 10

# -------------------------------------------------------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------------------------------------------------------

def ScoreSubmissions(avecs, svecs):
    """
    Compute cosine similarity between each new submission and each author abstract.
    Return dataframe sorted by this score.
    """

    # Compute cosine similarity
    all_scores = cosine_similarity(svecs, avecs)

    # For each submission, store index of author abstract that is most similar
    inds   = np.argmax(all_scores, axis=1)
    scores = np.max(all_scores, axis=1) 

    # Store this information in a sorted dataframe
    dd = {"score": scores, "author_index": inds}
    df = pd.DataFrame(data=dd)
    df.sort_values(by=["score"], ascending=False, inplace=True) 

    return df 

# -------------------------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------------------------

#
# Read in previously ranked submissions (will be empty on first pass)
#

submissions = Abstracts(past_sub_file, category=category)

#
# Update with the most recent submissions not yet ranked and preprocess their text
#

submissions.UpdateSubmissions(nsub_max)
submissions.TokenizeAbstracts()

#
# Rank the new submissions for each user
#

for author in authors:

    # Read author abstracts and preprocess text
    documents = Abstracts(input_file_temp % author)
    documents.TokenizeAbstracts()

    # Compute cosine similarity for each submission
    vectorizer = documents.FitVectorizer()
    avecs      = documents.RunVectorizer(vectorizer)
    svecs      = submissions.RunVectorizer(vectorizer)
    scores     = ScoreSubmissions(avecs, svecs)

    # Print information for the highest ranking submissions
    nprint = 0
    atitle = documents.GetTitles()
    stitle = submissions.GetTitles()
    sids   = submissions.GetIDs()
    sabs   = submissions.GetAbstracts()
    print(100*"-")
    print(("Top Ranked New Submissions for %s" % author).upper())
    print(100*"-")
    for index, row in scores.iterrows():
        print("ID              : %s" % sids[index])
        print("TITLE           : %s" % stitle[index])
        print("MOST SIMILAR TO : %s" % atitle[row["author_index"]])
        print("SCORE           : %f" % row["score"])
        print("ABSTRACT        : %s" % sabs[index])
        print()
        nprint += 1
        if nprint >= nprint_max: break

#
# Update data file with submissions ranked here 
#

submissions.DropTokens()
submissions.Save()

