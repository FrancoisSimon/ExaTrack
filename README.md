# ExaTrack

This repository contains the ExaTrack project which makes the syntesis between our previous tools ExTrack and aTrack. The goal of this project is to create an user-friendly tracking analysis tool able to account for multiple states of various types of motions and complex transitions between them. More precisely, this tool enable to account for immobile, diffusive, directed and confined motion and for transitions between these types of motion. This tool enables scientists to analyse their samples by characterizing the different types of motion and the transition kinetics, as well as labeling their tracks with state probabilities.

# Installation with anaconda
## To run ExaTrack on CPU with anaconda and spyder IDE:

`conda create -n PyExaTrack python=3.10.11`

`conda activate PyExaTrack`

`conda install spyder`

`pip install -r PATH\TO\requirements.txt`


## Running ExaTrack on GPU with anaconda and spyder IDE
`conda create -n PyExaTrack python=3.10.11`

`conda activate PyExaTrack`

`conda install spyder`

`conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`

`pip install -r PATH\TO\requirements.txt`

**WARNING**: On my hands, jit compilation fails with GPU computing. If using GPU parallelization, make sure to set `jit_compile=False` in the exatrack.py file.

You can then open spyder from the anaconda-navigator GUI or with the command line `spyder`


# License
This program is released under the GNU General Public License version 3 or upper (GPLv3+).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Author
Francois Simon
