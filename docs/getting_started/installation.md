# Installation

## Option A: Spyder standalone

Recommended for users with no Python background.

1. Install Spyder from [spyder-ide.org](https://www.spyder-ide.org/). Python is bundled.
2. Get the code: `git clone https://github.com/IRFM/D0FUS.git`, or
   **Code → Download ZIP** on [github.com/IRFM/D0FUS](https://github.com/IRFM/D0FUS) and extract it.
3. In Spyder, **File → Open…** `D0FUS.py`. The working directory is set to `D0FUS/` automatically.
4. In the IPython console: `%pip install -r requirements.txt`
   (use `!pip install -r requirements.txt` if `%pip` fails).
5. Press **F5** and select `D0FUS_INPUTS/1_run_ITER.txt` in the file picker.
   Results print to the console and figures open automatically.

## Option B: Miniforge (conda users)

```bash
conda create -n d0fus python=3.11 && conda activate d0fus
conda install pip spyder
git clone https://github.com/IRFM/D0FUS.git && cd D0FUS
pip install -r requirements.txt
python D0FUS.py D0FUS_INPUTS/1_run_ITER.txt
```

## Option C: pip (headless or library use)

```bash
pip install d0fus
```

The input decks of `D0FUS_INPUTS/` are not shipped in the PyPI package. Clone the
repository to get them.

## Optional 3D view

The 3D machine view uses PyVista:

```bash
pip install "d0fus[viz3d]"
```
