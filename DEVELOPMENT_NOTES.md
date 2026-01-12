# Short Term
* Round number in the scan mode -> Timothé
* Add a detailed output of the coil comprizing post-process of sigma_z, sigma_r, sigma_theta + Conductor shape and area for TF and CS -> Timothé
* Use classes as arguments of e.g. the run function instead of a long list of input and output variables -> Arthur
* Check pep8 conventions -> Arthur
* On LaTEx strings, use the string modifier "r" (e.g: r"$\alpha$") because, otherwise, "\[...]" indicates a special character in an unformatted text (currently it's a warning, but could lead to silent errors) -> Arthur will try to understand what this means
* Regarder utilisation lettres grecques dans python et décider si on enlève les lettres grecques -> Arthur
* Document the code using Sphinx ? It produces HTML documentation via formatted comments in the code (it's the most used documentation tool) -> Arthur
* It would be very nice to have an __init__.py file. That way, one could use the code as a library and not as a script. This would be an awesome thing to do because it reduces boilerplate. -> Arthur (à la fin)

# Mid Term
* Consider a distribution, even in Princeton D, where the voltage split is different from 50/50
* Evaluate the copper fraction needed from the stored energy and hot spot temperature and not as a fixed fraction
* Check the use of triangularity in the code, especially in V
* Alpha fraction to check
* Lineaic vs volumic density
* Psi calculation incoherent
* li poor calculation

# Long Term
* Go through all functions and make sure that no global variables are used
* Put all comments in English
* Put all variable names in English
* Use more adequate variable names, e.g. q_limit instead of q
* Give references for the models used, e.g. for the confinement time scaling laws