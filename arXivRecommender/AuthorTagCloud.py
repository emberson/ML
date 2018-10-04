import numpy as np
from Abstracts import Abstracts

# -------------------------------------------------------------------------------------------------------
# PARAMETERS
# -------------------------------------------------------------------------------------------------------

# Authors to construct tag clouds for 
authors = np.array(["emberson"])

# Input file containing author abstracts 
input_dir       = "data/"
input_file_temp = input_dir + "author-%s-ids.dat"

# Input file containing the corpus
input_corpus = input_dir + "corpus.dat"

# -------------------------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------------------------

for author in authors:

    #
    # Read author abstracts and the corpus
    #

    documents = Abstracts(input_file_temp % author) 
    corpus    = Abstracts(input_corpus)

    #
    # Clean text in both cases
    #

    documents.TokenizeAbstracts()
    corpus.TokenizeAbstracts()

    #
    # Compute TF-IDF weights on the author abstracts
    #

    vectorizer = corpus.FitCountVectorizer()
    tfidf = documents.ComputeWeights(vectorizer)

    tfidf.to_csv("data/tfidf_temp.dat", sep="\t")

