import numpy as np
from urllib.request import urlopen
import feedparser
import os.path

# -------------------------------------------------------------------------------------------------------
# PARAMETERS
# -------------------------------------------------------------------------------------------------------

# Authors to query
authors = np.array(["emberson"])

# Output file to save arXiv IDs
output_dir       = "data/"
output_file_temp = output_dir + "author-%s-ids.dat"

# -------------------------------------------------------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------------------------------------------------------

def RunAuthorQuery(aq):
    """
    Use the arXiv API to query a particular author. 
    See documentation of the arXiv API here: https://arxiv.org/help/api/index
    """ 

    # Query using the API
    url = "http://export.arxiv.org/api/query?search_query=au:" + aq + "&sortBy=submittedDate" 
    query = urlopen(url).read()

    return query

def UserSelectedResults(query):
    """
    Iterate through each author query to filter out incorrect results.
    """
    
    # Parse the Atom query 
    feed = feedparser.parse(query)

    # Iterate through each paper in the query and ask user wheter to add to the list
    ids = np.array([ ], dtype="|S200")
    for paper in feed.entries:
        title, authors, id = ParsePaper(paper)

        print("Title   : ", title)
        print("Authors : ", authors)
        print("ID      : ", id)
        keep = ParseInput(input(" >>> Do you want to keep this paper [Y/N] ? "))
        if keep:
            ids = np.append(ids, id)
            print(" >>> Adding to list")
        else:
            print(" >>> Not adding to list")
        print()

    return ids

def ParsePaper(paper):
    """
    Parse the Atom feed for this paper returning various attributes.
    """

    title = paper.title
    authorlist = paper.authors
    id = (paper.id.split("/")[-1]).split("v")[0]

    authors = ""
    for author in authorlist:
        authors += author["name"] + ", "
    authors = authors[:-2]

    return title, authors, id

def ParseInput(userinput):
    """
    Return bool specifying whether or not the user has specified to add this paper to the list.
    """

    keep = False
    if userinput.lower() == "y" or userinput.lower() == "yes": keep = True

    return keep

def SaveIDs(dw, ids):
    """
    Save selected IDs to a text file.
    """

    fw = open(dw, "w")
    for i in range(ids.shape[0]):
        fw.write(ids[i] + "\n")
    fw.close()

    print("arXiv IDs saved to " + dw)

# -------------------------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------------------------

if not os.path.isdir(output_dir): os.makedirs(output_dir)

for author in authors:

    print("Working on author : %s ..." % author)

    #
    # Run author query through the arXiv API
    #

    query = RunAuthorQuery(author)

    #
    # Interact with user to determine which search results are correct
    #

    ids = UserSelectedResults(query)

    #
    # Save selected IDs to text file
    #

    output_file = output_file_temp % author
    SaveIDs(output_file, ids)

    print("Done!")

