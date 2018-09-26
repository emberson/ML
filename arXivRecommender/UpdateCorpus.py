import numpy as np
from Abstracts import Abstracts
import os.path

# -------------------------------------------------------------------------------------------------------
# PARAMETERS
# -------------------------------------------------------------------------------------------------------

# arXiv category to build corpus from
category = "astro-ph.CO"

# Number of most recent abstracts to base corpus on.
ncorpus = 15000

# Output files containing corpus and list of ids it contains
output_dir  = "data/"
output_file = output_dir + "corpus.dat" 

# -------------------------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------------------------

#
# Make output directory if does not yet exist
#

if not os.path.isdir(output_dir): os.makedirs(output_dir)

#
# Instantiate abstract class with output file and corpus category 
#

abstracts = Abstracts(output_file, category=category)

#
# Either build corpus from scratch or update it with most recent abstracts 
#

abstracts.UpdateCorpus(ncorpus)

#
# Save dataframe to output file
#

abstracts.Save()

