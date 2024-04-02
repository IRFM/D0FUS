<a name="readme-top"></a>

# D0FUS

## Getting Started

The code is currently presented as a single Python file comprising a tree of functions allowing for multiple objectives to be achieved:

* A first initialization phase is necessary with the import of libraries, the definition of constants, the creation of a database of main fusion machines, and the initialization of numerical parameters.
* Then comes the definition of physical and engineering functions, the core of the program, although not the largest part of the code.
* Other functions, for routine tasks, calculations, and graphical display, are then defined.
* Finally, modules allowing the production of exploitable results are presented:
  - Variation of the plasma radius for a given set of input data
  - Variation of H, B_max, P_fus, P_w, or f_obj for the rest of the given data
  - A gradient descent to find the minimum of an arbitrary cost function while keeping the input values of the code between user-selected limits
  - A robustness test to calculate the outputs of D0FUS for a set of input data, with aesthetic displays
  - Finally, the possibility to plot a variation of R_0 as a function of B for different assumptions.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Prerequisites

D0FUS is a code developed in Python. To open it, the author recommends using Spyder, but any other Python executor will suffice. The code uses the following librarys, which will need to be installed on first execution if not already present:

* numpy
* matplotlib
* scipy
* pandas
* warnings
* os
* plotly
* seaborn
* math

To do this, if necessary, execute in the console:

* CMD
  ```sh
  pip install library_name
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Installation

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
  git clone http://irfm-gitlab.intra.cea.fr/TA276941/D0FUS.git
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Developers

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
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Outside Contributers

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

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Server

For users belonging to IRFM, execution on the IRFM JupyterLab server is also possible and convenient. To do this, go to the IRFM website, then navigate to the 'Pratique' tab, and then 'Python IRFM'.

## Roadmap

- [ ]  Passage sous GIT
    - [x]  Transfert Eric et Yanick
    - [ ]  Documents ressources à créer

- [ ]  Débugage massif :
    - [x]  fb/fnc à fraction recirculante
    - [ ]  Pfus à Pe , check ce que ça veut dire + bout de note ?
    - [ ]  Descente de gradient
    - [ ]  RiCS
    - [ ]  Test de robustesse
    - [ ]  R/B

- [ ]  Changement lois d’échelles (Verdolaje 2022)
- [ ]  Outil de scan avec base de donnée interactive

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Auclair Timothe - timothe.auclair@cea.fr

Project Link: [http://irfm-gitlab.intra.cea.fr/TA276941/D0FUS.git]

<p align="right">(<a href="#readme-top">back to top</a>)</p>
