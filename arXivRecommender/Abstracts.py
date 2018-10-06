import numpy as np
import pandas as pd
from urllib.request import urlopen
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import feedparser
import string
import os.path
import time

class Abstracts:
    """
    Class that holds a collection of abstracts and their corresponding arXiv ids either for the
    entire corpus or for an individual author. In either case, this class contains the functionality
    to update the collection with new arXiv submissions. Natural language processing algorithms are
    run directly on this class to compute term and inverse document frequency.
    """

    # Column names in pandas dataframe
    id = "id"
    abstract = "abstract"
    cleaned  = "cleaned"

    # Default arXiv corpus category
    arxiv_category = "astro-ph.CO"

    # Some default flags
    automatic_author_update = False
    UseStemming = False

    def __init__(self, data_file, category=None):
        """
        Instantiate class with the input/output file containing the pandas dataframe. 
        If this file exists, read its contents; otherwise, construct an empty dataframe. 
        """

        # File which stores the dataframe
        self.data_file = data_file

        # Update corpus category if provided
        if category is not None: self.arxiv_category = category

        # Either read in existing dataframe or create an empty one
        if os.path.isfile(self.data_file):
            self.Read()
        else:
            self.collection = pd.DataFrame(columns=[self.id, self.abstract])

    def AutomaticAuthorUpdate(self, flag):
        """
        Set the value of the flag controlling automatic author updating.
        """

        self.automatic_author_update = flag

    def BuildFromAuthorQuery(self, author):
        """
        Build the collection by running an author query on the arXiv API. Interact with the
        user to select which results from the query actually pertain to the author at hand.
        Alternatively, if automatic_author_update is set then all results from the query will
        be added for the author (potentially risky, but necessary if one wants to update the
        author list with a cron job for instance).
        """ 

        # Query using the arXiv API
        url = "http://export.arxiv.org/api/query?search_query=au:" + author + "&sortBy=submittedDate"
        query = urlopen(url).read()

        # Parse the Atom query 
        feed = feedparser.parse(query)

        # Iterate through each paper in the query and (optionally) ask user wheter to add to the list
        for paper in feed.entries:

            AddAbstract = False
            title, authors, id, abstract = self.ParsePaper(paper)

            # Only continue if this paper is not already in the list
            if self.CheckID(id):
                print("Skipping already included abstract %s" % id)
                continue 
            
            # Either automatically add this paper or ask user for permission
            if self.automatic_author_update:
                AddAbstract = True
            else:
                print("Title   : ", title)
                print("Authors : ", authors)
                print("ID      : ", id)
                AddAbstract = self.ParseUserInput(input(((" >>> Do you want to add this paper to author %s [y/n] ? ") % author).upper()))

            # Add abstract if desired
            if AddAbstract: self.AddAbstract(id, abstract, Verbose=True)

        # Sort so that most recent paper comes first
        self.SortDescending()

    def UpdateCorpus(self, ncorpus, nabs_per_iter=100, wait_time=3):
        """
        Update corpus so that it contains the ncorpus most recent abstracts in the arxiv category.
        """

        #
        # Make sure collection is sorted with most recent first so that we know when to stop
        #

        self.SortDescending()

        #
        # Page through arXiv query results (see https://arxiv.org/help/api/examples/python_arXiv_paging_example.txt)
        #

        # Template URL query that sorts by submission date
        url_template = "http://export.arxiv.org/api/query?search_query=cat:"+self.arxiv_category+"&start=%i&max_results=%i&sortBy=submittedDate"

        # Keep track of how many new abstracts have been added
        nadd = 0

        for i in range(0, ncorpus, nabs_per_iter):

            # Query the arXiv API for search results within specifix document range
            url = url_template % (i, nabs_per_iter)
            query = urlopen(url).read()

            # Store abstract of each new paper and stop if we have already added it 
            feed = feedparser.parse(query)
            StopSearch = False
            for paper in feed.entries:
                title, authors, id, abstract = self.ParsePaper(paper)
                if self.CheckID(id):
                    StopSearch = True
                    break
                self.AddAbstract(id, abstract, Verbose=False)
                nadd += 1
            if StopSearch: break

            # Friendly wait before querying the API again 
            time.sleep(wait_time)

        #
        # Trim so that only the ncorpus most recent abstracts are included 
        #

        self.TrimCollection(ncorpus)

        #
        # Print some info to screen
        #

        print()
        print("Corpus contains %i abstracts" % self.collection.shape[0])
        print("Abstracts added now : %s" % nadd)
        print("Newest arxiv id     : %s" % self.collection[self.id].iloc[0])
        print("Oldest arxiv id     : %s" % self.collection[self.id].iloc[-1])
        print()

    def AddAbstract(self, id, abstract, Verbose=False):
        """
        Add the provided abstract to the dataframe.
        """

        self.collection = self.collection.append({self.id:id, self.abstract:abstract}, ignore_index=True)
        if Verbose: print("Added abstract with id : %s" % id)

    def CheckID(self, id):
        """
        Check arxiv id to see if this abstract is already in the collection.
        """

        contained = False
        if self.collection[self.id].str.contains(id).sum(): contained = True
        return contained

    def SortDescending(self):
        """
        Sort the dataframe in descending order of arxiv id (makes most recent first)
        """

        self.collection.sort_values(by=[self.id], ascending=False, inplace=True)
        self.collection.reset_index(drop=True, inplace=True)

    def TrimCollection(self, n):
        """
        Trim so that only the n most recent abstracts are in the collection.
        """
    
        # First make sure most recent are first
        self.SortDescending()

        # Now keep only n most recent
        self.collection = self.collection[:n]

    def ParsePaper(self, paper):
        """
        Parse the Atom feed for this paper returning various attributes.
        """

        title = paper.title
        authorlist = paper.authors
        id = str((paper.id.split("/")[-1]).split("v")[0])
        abstract = paper.summary

        authors = ""
        for author in authorlist:
            authors += author["name"] + ", "
        authors = authors[:-2]

        return title, authors, id, abstract

    def ParseUserInput(self, userinput):
        """
        Return bool specifying whether or not the user has specified to add this paper to the list.
        """

        keep = False
        if userinput.lower() == "y" or userinput.lower() == "yes": keep = True

        return keep

    def Save(self):
        """
        Write dataframe to output file.
        """

        self.collection.to_csv(self.data_file, sep="\t")
        print("Abstracts saved to " + self.data_file)

    def Read(self):
        """
        Read previously saved input file.
        """

        self.collection = pd.read_csv(self.data_file, index_col=0, dtype=str, sep="\t")
        print("Abstracts read from " + self.data_file)

    def TokenizeAbstracts(self):
        """
        Tokenize the text in each abstract to compute uni-, bi-, and trigrams. First clean the
        text by removing punctuation, converting to lowercase, and removing numeric, short, and stop words. 
        """

        #
        # Add new column to dataframe to store cleaned text
        #
        
        self.collection.insert(self.collection.shape[1], self.cleaned, "")
    
        #
        # Clean abstract one row at a time 
        #

        for index, row in self.collection.iterrows():

            # Grab abstract
            abstract = row[self.abstract]

            # Separate hyphenated words
            abstract = abstract.replace("-", " ")

            # Convert to lowercase words
            words = word_tokenize(abstract)
            words = [w.lower() for w in words]

            # Remove punctuation
            table = str.maketrans("", "", string.punctuation)
            words = [w.translate(table) for w in words]

            # Remove numeric words
            words = [w for w in words if w.isalpha()] 

            # Remove stop words
            stop_words = set(stopwords.words("english"))
            words = [w for w in words if w not in stop_words]
        
            # Remove short words
            words = [w for w in words if len(w) > 2]

            # Stem words
            if self.UseStemming:
                stemmer = SnowballStemmer("english")
                words = [stemmer.stem(w) for w in words]

            # Place cleaned data in appropriate column
            cleaned_text = ""
            for w in words: cleaned_text += w + " "
            row[self.cleaned] = cleaned_text

        print("Cleaned and tokenized abstracts")

    def ComputeTfidfWeights(self):
        """
        Compute TF-IDF weights using the cleaned and tokenized abstracts.
        """

        # Store cleaned text as a list 
        doc_text = [ ]
        for index, row in self.collection.iterrows(): doc_text.append(row[self.cleaned])

        vectorizer = TfidfVectorizer(ngram_range=(1,3), norm="l2")
        weights = vectorizer.fit_transform(doc_text)
        weights = np.asarray(weights.mean(axis=0)).ravel().tolist()
        weights = pd.DataFrame({"term": vectorizer.get_feature_names(), "weights": weights})
        weights.sort_values(by="weights", ascending=False, inplace=True)
        weights.reset_index(drop=True, inplace=True)
    
        return weights

