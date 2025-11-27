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

Comments from Sebastian Kay Conde Lorenzo: 
* On LaTEx strings, use the string modifier "r" (e.g: r"$\alpha$") because, otherwise, "\[...]" indicates a special character in an unformatted text (currently it's a warning, but could lead to silent errors).
* A requirements file would be nice, because then you don't have to install manually all the packages (plus you don't have compatibility issues. I'll send one I generated, but you can do it by installing pipreqs and then running "python3 -m pipreqs.pipreqs.)
* Regarder utilisation lettres grecques dans python ?
* (It would be very nice to have an __init__.py file and more modularization. That way, one could use the code as a library and not as a script! This would be an awesome thing to do because it reduces boilerplate.)
* Finally, I suggest documenting the code using Sphinx. It produces HTML documentation via formatted comments in the code (it's the most used documentation tool). If you intend to open source the project, then this would be very helpful because then end-users (physicists that don't want/like to code, for example) could use the library without having to dig in the code and all of that. I have an example in a project of mine, I'll link it here https://github.com/cbasitodx/Minitorch

# For discussion

* Remove D0FUS_2D_Scan.py and use only D0FUS_2D_Scan_generic.py (and rename it D0FUS_2D_Scan.py)?
* Remove Graphs directories?
* Remove capsules at the top of each file giving the date of first creation and the author?