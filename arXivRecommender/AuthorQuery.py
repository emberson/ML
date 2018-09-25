import numpy as np
from Abstracts import Abstracts 
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
# MAIN
# -------------------------------------------------------------------------------------------------------

# Make output directory if does not yet exist
if not os.path.isdir(output_dir): os.makedirs(output_dir)

for author in authors:

    print("Working on author : %s ..." % author)

    # Instantiate abstract class with output file
    abstracts = Abstracts(output_file_temp % author)

    # Update author collection with arXiv author search
    abstracts.BuildFromAuthorQuery(author)
    
    # Save dataframe to output file
    abstracts.Save()

    print("Done!")

