# To do

* Put all comments in English
* Put all variable names in English
* Avoid code duplication for the printing of results (currently duplicated in D0FUS_run and D0FUS_Scan_2D_generic)
* In D0FUS_Scan_2D_generic, instead of creating many matrices "by hand", can we not create a matrix of (class) Output_parameters? This would save a lot of lines of code.
* In D0FUS_Scan_2D_generic, allow users to choose any couple of output parameters for iso-contours
* Move D0FUS_run.py into D0FUS instead of D0FUS/D0FUS_BIB
* Put more explicit comments in D0FUS_parameterization.py, e.g. for ITERP -> Timothé
* Add legend in 2D scan plots for the {radial build + plasma stability} contour
* Use classes as arguments of e.g. the run function instead of a long list of input and output variables
* Go through all functions and make sure that no global variables are used
* Change the code structure to remove the imports in cascade
* Use more adequate variable names, e.g. q_limit instead of q 
* Give references for the models used, e.g. for the confinement time scaling laws -> Timothé
* Remove function in function
* only 1  __name__ == '__main__' per file

# For discussion

* Remove D0FUS_2D_Scan.py and use only D0FUS_2D_Scan_generic.py (and rename it D0FUS_2D_Scan.py)?
* Remove Graphs directories?
* Remove capsules at the top of each file giving the date of first creation and the author?