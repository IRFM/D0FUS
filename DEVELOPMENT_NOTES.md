# To do

* Add a detailed output of the coil comprizing post-process of sigma_z, sigma_r, sigma_theta + Conductor shape and area for TF and CS -> Timothé
* In D0FUS_Scan_2D_generic, instead of creating many matrices "by hand", can we not create a matrix of (class) Output_parameters? -> Timothé
* In D0FUS_Scan_2D_generic, allow users to choose any couple of output parameters for iso-contours -> Timothé
* Use classes as arguments of e.g. the run function instead of a long list of input and output variables -> Arthur
* Check pep8 conventions -> Arthur
* Comment from Sebastian Kay Conde Lorenzo: On LaTEx strings, use the string modifier "r" (e.g: r"$\alpha$") because, otherwise, "\[...]" indicates a special character in an unformatted text (currently it's a warning, but could lead to silent errors). -> Arthur will try to understand what this means
* Regarder utilisation lettres grecques dans python et décider si on enlève les lettres grecques -> Arthur
* Comment from Sebastian Kay Conde Lorenzo: Document the code using Sphinx? It produces HTML documentation via formatted comments in the code (it's the most used documentation tool). If you intend to open source the project, then this would be very helpful because then end-users (physicists that don't want/like to code, for example) could use the library without having to dig in the code and all of that. I have an example in a project of mine, I'll link it here https://github.com/cbasitodx/Minitorch -> Arthur
* It would be very nice to have an __init__.py file and more modularization. That way, one could use the code as a library and not as a script! This would be an awesome thing to do because it reduces boilerplate. -> Arthur (à la fin)
* Perhaps we should consider a distribution, even in Princeton D, where the voltage split is different from 50/50.

# General rules of conduct

* Put more explicit comments in D0FUS_parameterization.py, e.g. for ITERP -> Timothé
* Go through all functions and make sure that no global variables are used -> Timothé
* Put all comments in English -> Timothé
* Put all variable names in English -> Timothé
* Use more adequate variable names, e.g. q_limit instead of q 
* Give references for the models used, e.g. for the confinement time scaling laws