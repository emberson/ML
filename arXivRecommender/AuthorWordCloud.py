import numpy as np
from Abstracts import Abstracts
from wordcloud import WordCloud
import matplotlib
matplotlib.use("PDF")
import pylab
import os

import pandas as pd

# -------------------------------------------------------------------------------------------------------
# PARAMETERS
# -------------------------------------------------------------------------------------------------------

# Authors to construct word clouds for 
authors = np.array(["emberson"])

# Input file containing author abstracts 
input_dir       = "data/"
input_file_temp = input_dir + "author-%s-ids.dat"

# Output plot containing the word cloud
plot_dir       = "plots/"
plot_file_temp = plot_dir + "wordcloud-%s.pdf"

# Path to non-default font if do not want to use wordcloud default
font_path = "path/to/nonstandard/font.ttf"

# Path to non-default colormap if do not want to use matplotlib ones
cmap_path = "path/to/nonstandard/colormap"

# -------------------------------------------------------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------------------------------------------------------

def ComputeTermFrequencies(w, n):
    """
    Return a dictionary with weight mappings from the top n terms.
    """

    f = { }
    for index, row in w[:n].iterrows(): 
        f[row["term"]] = row["weights"]

    return f

def ReadPalette(data_file, N=256):
    """
    Read RGB palette and convert to matplotlib colormap.
    """

    fr = open(data_file, "r")
    lines = fr.readlines()
    fr.close()

    r = np.zeros(shape=(N, 3), dtype="float32")
    g = np.zeros(shape=(N, 3), dtype="float32")
    b = np.zeros(shape=(N, 3), dtype="float32")

    ind = np.linspace(0., 1., N)

    for i in range(len(lines)):

        cline = str.split(lines[i])

        r[i][0] = ind[i]
        g[i][0] = ind[i]
        b[i][0] = ind[i]

        r[i][1:] = float(cline[0])
        g[i][1:] = float(cline[1])
        b[i][1:] = float(cline[2])

    r = tuple(tuple(x) for x in r)
    g = tuple(tuple(x) for x in g)
    b = tuple(tuple(x) for x in b)

    # Convert this to matplotlib colormap
    cdict = {"red": r, "green": g, "blue": b}
    cmap = matplotlib.colors.LinearSegmentedColormap("colormap", cdict, N)

    return cmap

def PlotWordCloud(pf, w, font_path, cmap_path, cmap="jet", max_terms=64):
    """
    Plot author word cloud.
    """

    #
    # Check if user has supplied non-default font or colormap paths.
    # Otherwise set these to None which will use wordcloud defaults.
    #

    if not os.path.isfile(font_path): font_path = None
    if os.path.isfile(cmap_path): cmap = ReadPalette(cmap_path)
    else: cmap = None

    #
    # Use tf-idf score to define frequencies for word cloud
    #
    
    frequencies = ComputeTermFrequencies(w, max_terms)

    #
    # Setup the plot
    #

    fig_width     = 1280
    fig_height    = 720
    inches_per_pt = 1. / 72.27
    pylab.rcParams.update({"figure.figsize":[fig_width*inches_per_pt, fig_height*inches_per_pt]})
    pylab.figure(1)
    pylab.clf()

    #
    # Make elliptical mask
    #

    x, y = np.ogrid[:fig_height, :fig_width]
    mask = (2*(x-(fig_height/2))/fig_height)**2 + (2*(y-(fig_width/2))/fig_width)**2 > 0.98
    mask = 255 * mask.astype(int)

    #
    # Generate word cloud with this mask
    #

    wordcloud = WordCloud(font_path=font_path, margin=0, background_color="white", colormap=cmap, min_font_size=8, mask=mask)
    wordcloud = wordcloud.generate_from_frequencies(frequencies)

    #
    # Plot with pylab
    #

    pylab.imshow(wordcloud, interpolation='bilinear')

    pylab.axis("off")
    pylab.subplots_adjust(left=0, right=1, bottom=0, top=1)

    pylab.savefig(pf)
    pylab.close()

    print("Plot saved to " + pf)

# -------------------------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------------------------

# Make output directory if does not yet exist
if not os.path.isdir(plot_dir): os.makedirs(plot_dir)

for author in authors:

    # Read author abstracts
    documents = Abstracts(input_file_temp % author) 

    # Clean text and tokenize
    documents.TokenizeAbstracts()

    # Compute TF-IDF weights on the tokenized text 
    tfidf = documents.ComputeTfidfWeights()

    # Plot the word cloud
    PlotWordCloud(plot_file_temp % author, tfidf, font_path, cmap_path)

