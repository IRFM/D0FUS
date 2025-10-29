# To do

* Set a fixed number of ticks (10 by default) in the 2D scan plots -> Arthur
* Improve output files from D0FUS_2D_Scan_generic -> Eric
* Put all comments in English
* Put all variable names in English
* Move D0FUS_run.py into D0FUS instead of D0FUS/D0FUS_BIB
* Put more explicit comments in D0FUS_parameterization.py, e.g. for ITERP -> Timothé
* Add legend in 2D scan plots for the radial build + plasma stability contour
* Use classes as arguments of e.g. run instead of a long list of input and output variables
* Go through all functions and make sure that no global variables are used
* Change the code structure to remove the imports in cascade

# For discussion

* Remove D0FUS_2D_Scan.py and use only D0FUS_2D_Scan_generic.py (and rename it D0FUS_2D_Scan.py)?
* Remove Graphs directories?
* Remove capsules at the top of each file giving the date of first creation and the author?