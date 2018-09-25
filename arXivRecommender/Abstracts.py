import numpy as np
import pandas as pd
from urllib.request import urlopen
import feedparser
import os.path

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

    # Some default flags
    automatic_author_update = False

    def __init__(self, data_file):
        """
        Instantiate class with the input/output file containing the pandas dataframe. 
        If this file exists, read its contents; otherwise, construct an empty dataframe. 
        """

        # File which stores the dataframe
        self.data_file = data_file

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

            AddPaper = False
            title, authors, id, abstract = self.ParsePaper(paper)

            # Only continue if this paper is not already in the list
            if self.collection[self.id].str.contains(id).sum():
                continue
            
            # Either automatically add this paper or ask user for permission
            if self.automatic_author_update:
                AddPaper = True
            else:
                print("Title   : ", title)
                print("Authors : ", authors)
                print("ID      : ", id)
                AddPaper = self.ParseUserInput(input(((" >>> Do you want to add this paper to author %s [y/n] ? ") % author).upper()))
            if AddPaper:
                self.collection = self.collection.append({self.id:id, self.abstract:abstract}, ignore_index=True)
                print(" >>> Added to list ID : %s" % id)

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

