This repository contains the custom extension to the LAMMPS code and an example LAMMPS script to accompany the publication 'Recrystallization in String Fluid Complex Plasmas'. Please cite the paper if you use this code.
You may find the data generated by this code here: https://zenodo.org/record/5871983 .

LAMMPS - a flexible simulation tool for particle-based materials modeling at the atomic, meso, and continuum scales, A. P. Thompson, H. M. Aktulga, R. Berger, D. S. Bolintineanu, W. M. Brown, P. S. Crozier, P. J. in 't Veld, A. Kohlmeyer, S. G. Moore, T. D. Nguyen, R. Shan, M. J. Stevens, J. Tranchida, C. Trott, S. J. Plimpton, Comp Phys Comm, 271 (2022) 10817.  lammps.sandia.gov/

The work was performed by Dr. Mierk Schwabe and Eshita Joshi at Deutsches Zentrum für Luft- und Raumfahrt.
Please contact Eshita Joshi at eshita.joshi@dlr.de for further questions.

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

This code can be used to generate string fluids and crystals in complex plasmas.

The following files are provided:

	1. example_fluid.restart: This is a LAMMPS binary restart file with the saved state of an example fluid at thermodynamic equilibrium.
	2. example_string.in: This is a LAMMPS input file used as an example to illustrate the use of the custom LAMMPS interparticle potential.
	3. pair_coul_string.cpp: This is the custom C++ file used to add the 'coul/string' option to the 'pair_style' LAMMPS command to compute interparticle potential.
	4. pair_coul_string.h: This is the header file accompanying the C++ extension.

How to run:

	1. Add the pair_coul_string.cpp and pair_coul_string.h files to the /mylammps/src/ folder
	2. Compile LAMMPS the usual way
	3. Run example_string.in using LAMMPS the usual way

The following parameters can be changed in the input file and have been marked as such:

	1. Wake charge: Increasing the wake charge increases the strength of the interparticle force
	2. Wake distance: Increasing the wake distance increases the strength of the interparticle force
	3. Pressure: Increasing pressure increases damping due to gas friction
	4. Temperature: Increasing temperature increases the kinetic energy of the particles