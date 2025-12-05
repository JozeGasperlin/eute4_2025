# Code for "Room-temperature memristive switching between charge density wave states" (2025)

We provide the minimal <code>Python</code> code required to reproduce the numerical results presented in our publication.

The <code>lib</code> folder contains the backbone of the numerical implementation for the model presented in the "Methods" section:
- k-space, Hamiltonian and density matrix array constructors along with relevant functions in <code>arrays.py</code>
- derivatives and update functions for the Hellmann-Feynman EOM-based optimisation procedure in <code>eom.py</code>
- computing the filling, chemical potential and energies in <code>observables.py</code>
- convenience functions in <code>auxiliary.py</code>

The <code>scripts</code> folder contains a number of scripts using <code>lib</code> to compute the presented data:
- <code>find_gnd.py</code> uses the EOM optimisation procedure to find a stable configuration given an initial guess of $(\alpha, \theta_x, \theta_y)$ for each of $L$ layers of an $L$-layer system,
- <code>dispersion.py</code> computes the electronic dispersion given a set of parameters as above,
- <code>energy_surface_cut.py</code> computes the energy of a series of parameter sets leading to a cut through the energy surface between two given configurations (see Fig. 4a),
- <code>lindhard_bilayer.py</code> computes the Lindhard response of a hybridised bilayer in a given window in $q$-space (see Figs. 4d, 4e).

The <code>notebooks</code> folder contains a series of Jupyter notebooks, where we provide the code reproducing the Figures used in the article:
- <code>lattice_and_unit_cells.ipynb</code>  for Figs. 1b, 4b, 4c and S12.
- <code>lindhard.ipynb</code> for Fig. S14 and Figs. 4d, 4e. For the latter, one needs to first generate the data using <code>lindhard_bilayer.py</code>.
- <code>energy_surface_and_dispersion.ipynb</code> for Figs. 4a and S13. Here one needs to produce the data using <code>energy_surface_cut.py</code> and <code>dispersion.py</code>, respectively.

The raw data obtained from the numerical simulations is also contained in the <code>data</code> folder.

The module versions used at time of publication were:

- <code>python 3.14</code>
- <code>numpy 2.3.5</code>
- <code>scipy 1.16.3</code>
- <code>matplotlib 3.10.8</code>
- <code>tqdm 4.67.1</code> (optional but nice)