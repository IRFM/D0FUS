<a name="readme-top"></a>

# D0FUS

## Getting Started

The code is currently presented as a Python library comprising a tree of .py scripts for multiple objectives:
* D0FUS_import: Import all the necessary libraries
* D0FUS_parametrization: Panel control of the code, with the definition of all the constants
* D0FUS_physical_functions: Definition of all the physical functions
* D0FUS_radial_build_functions: Definition of the functions allowing for the different width calculations
* D0FUS_run: Major function definition, allowing for the calculation of a complete design point
* D0FUS_plot: Aesthetic functions

Additionally, the 'main' files, utilizing the previously presented library, are representative of the possible and current uses of the code:
* D0FUS_Benchmark: Provides a benchmark with several machines
* D0FUS_Gradient: Enables the search for an optimized point using various gradient descent techniques (mainly genetic algorithms)
* D0FUS_Scan_1D: 1D scan (typically on major or minor radius)
* D0FUS_Scan_2D: 2D scan (typically on major and minor radius). Major use of the code and the most refined script. Allows for complex visualization, choosing between several parameter plots.
* D0FUS_Scan_3D: 3D scan (typically on major and minor radius + magnetic field)

Please note that currently, not all the 'main' files are fully operational.

Additionally, and for more details, useful papers and notes on the code are grouped in the 'Reference Papers' file:
* [Plasma Physics Paper](http://irfm-gitlab.intra.cea.fr/projets/pepr-suprafusion/D0FUS/-/blob/main/Reference%20Papers/D0FUS_Auclair_Timothe_FED_Submission_29.11.24.pdf)
* [Radial Build Paper](http://irfm-gitlab.intra.cea.fr/projets/pepr-suprafusion/D0FUS/-/blob/main/Reference%20Papers/D0FUS_Auclair_Timothe_Radial_Build_09.01.2024.pdf?ref_type=heads)


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Prerequisites

D0FUS is a code fully developed in Python.
To use it, the author recommends using Spyder, but any other Python executor will suffice.
The code uses the following librarys:

* os
* math
* time
* warnings
* sys
* importlib
* numpy
* pandas
* sympy
* scipy
* matplotlib
* mpl_toolkits
* tqdm

To do this, if necessary, execute in the console:

* CMD
  ```sh
  conda install library_name
  ```

If you want to install every librarys in one single command :

* CMD
  ```sh
  conda install os math time warnings sys importlib numpy pandas sympy scipy matplotlib tqdm

<p align="right">(<a href="#readme-top">back to top</a>)</p>

Attntion : conda install might not work, then use pip install.

In practice, most of the library are pre-installed in most Python executor, at the exception of tqdm.

## Installation

**Classical :**

To create a local copy of this project to use it, here are the necessary steps:

1. Download GIT on you machine
2. Open GIT BASH where you want to clone the project
3. Identification :
  ```sh
  git config --global user.name USERNAME
  ```
  ```sh
  git config --global user.email EMAIL
  ```
4. Clone the project :
  ```sh
  git clone http://irfm-gitlab.intra.cea.fr/projets/pepr-suprafusion/D0FUS.git
  ```
**Alternative :**

Download all the folders from the git repository

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Developers

**Classical :**

To update the main branch, here are the steps to follow:
1. Add the necessary files to the GIT cache :
  ```sh
  git add FICHIER.TYPE
  ```
2. Commit the necessary files to the local GIT repository :
  ```sh
  git commit -m 'Actualisation message'
  ```
3. Push the local repository to the server:
  ```sh
  git push HEAD:main
  ```
**Alternative :**

Modify the code in the GIT interface and save as a new commit with associated message.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Outside Contributers

**Classical :**

If you have a suggestion that would make this better, please fork the repository and create a pull request. To do so :

1. Fork the Project
2. Create your Feature Branch 
```sh
git checkout -b feature/AmazingFeature
```
3. Commit your Changes
```sh
git commit -m 'Add some AmazingFeature'
```
4. Push to the Branch
```sh
git push origin feature/AmazingFeature
```
5. Open a Pull Request

You can also open an issue with the tag "enhancement".

**Alternative :**

From the GIT user interface create a new Branch.
Modify the code in the GIT interface and save as a new commit with associated message.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Server

For users belonging to IRFM, execution on the IRFM JupyterLab server is also possible and convenient. To do so, go to the IRFM website, then navigate to the 'Pratique' tab, and then 'Python IRFM'.

## License

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Auclair Timothe - timothe.auclair@cea.fr

Project Link: [http://irfm-gitlab.intra.cea.fr/projets/pepr-suprafusion/D0FUS/.git]

<p align="right">(<a href="#readme-top">back to top</a>)</p>
