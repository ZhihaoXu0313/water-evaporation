from lammps import lammps
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from sklearn.cluster import DBSCAN

from utils import *


########################################################################
# load lammps objective
lmp               = lammps()

# setup simulation
thermo            = 1000                      # thermo info
timestep          = 0.1                       # timestep

NVE_1_steps       = 200000                    # NVE step
NVT_1_steps       = 500000                    # NVT step
NVT_2_steps       = 1000000                   # ENVT step

n_bins            = 525                       # number of bins on the z-direction
n_grid_x          = 10                        # mesh dim x-direction
n_grid_y          = 10                        # mesh dim y-direction

n_steps           = 2000                      # update freq for E field
Elow              = 0                         # E strength (low)
Ehigh             = 1e-1                      # E strength (high)
E                 = Ehigh - Elow
Efreq             = 6e14                      # AC E field freq
k                 = 20.0                      # shape constant

profile_name      = "density_profile_eq.dat"  # file to save density profile
surface_directory = "./surface"               # surface file folder
########################################################################

# start simulation
init_script = f"""
units        real
atom_style   full
atom_modify  map array
read_data    tip5p.data

mass         1 15.9994
mass         2 1.008
mass         3 1.0e-100

pair_style   lj/cut/coul/cut 8.0
pair_coeff   1 1 0.160  3.12
pair_coeff   2 2 0.0    1.0
pair_coeff   3 3 0.0    1.0

thermo_style custom step temp press etotal density pe ke
thermo       {thermo}
timestep     {timestep}
neigh_modify exclude molecule/intra all

fix          bottom all wall/lj126 zlo EDGE 2.0 1.0 2.5 pbc yes
fix          top all wall/lj126 zhi EDGE 2.0 1.0 2.5 pbc yes
"""
lmp.commands_list(init_script.splitlines())
########################################################################

# NVE
nve_script = f"""
fix          integrate_nve all rigid/nve/small molecule
dump         1 all custom 1000 nve.lammpstrj id type x y z
run          {NVE_1_steps}
unfix        integrate_nve
undump       1
reset_timestep 0
"""
lmp.commands_list(nve_script.splitlines())

# NVT
density_profile = []

nvt_1_script = f"""
compute      Chunks_equ all chunk/atom bin/1d z lower 0.2 units box
fix          Density_equ all ave/chunk 1000 1 1000 Chunks_equ density/mass file {profile_name}

fix          integrate_nvt all rigid/nvt/small molecule temp 350.0 350.0 100.0
dump         2 all custom 1000 nvt.lammpstrj id type x y z
"""
lmp.commands_list(nvt_1_script.splitlines())
lmp.command(f"run          {NVT_1_steps}")
lmp.command("unfix          integrate_nvt")
lmp.command("unfix          Density_equ")
lmp.command("undump         2")

################################## if plot density
with open(profile_name, 'r') as dfile:
    lines = dfile.readlines()[3:]
density = np.zeros((len(lines) // (n_bins + 1), n_bins, 4))
for i in range(len(lines) // (n_bins + 1)):
    tmp = lines[i * (n_bins + 1) + 1: i * (n_bins + 1) + n_bins + 1]
    density[i,:,:] = np.array([t.strip().split() for t in tmp])
ave_density = np.mean(density, axis=0)

plt.scatter(ave_density[:, 1], ave_density[:, 3])
plt.xlabel("z (Angstrom)")
plt.ylabel("Density (g/cm^3)")
plt.ylim([0, 1.5])
plt.savefig("density_profile_relax.png", dpi=300)

################################## if plot surface
init_coord = lmp.numpy.extract_atom('x')
init_box = lmp.extract_box()

xlo, xhi = init_box[0][0], init_box[1][0]
ylo, yhi = init_box[0][1], init_box[1][1]
zlo, zhi = init_box[0][2], init_box[1][2]

print("Coordinates: ", init_coord)

dx = (xhi - xlo) / n_grid_x
dy = (yhi - ylo) / n_grid_y

surface_landscape = np.zeros((n_grid_x, n_grid_y))

for i in range(n_grid_x):
    xrange = [i * dx, (i + 1) * dx]
    for j in range(n_grid_y):
        yrange = [j * dy, (j + 1) * dy]
        print("x-y block: ", xrange, yrange)
        grid_mols = init_coord[(init_coord[:, 0] > xrange[0]) & 
                               (init_coord[:, 0] < xrange[1]) & 
                               (init_coord[:, 1] > yrange[0]) & 
                               (init_coord[:, 1] < yrange[1])]
        surface_landscape[i, j] = identify_surface(grid_mols, zlo, zhi)

np.savetxt("init_surface.txt", surface_landscape, delimiter=' ')
plot_surface(filename=f"{surface_directory}/init_surface.png", 
             surface_landscape=surface_landscape,
             xlo=xlo, xhi=xhi, 
             ylo=ylo, yhi=yhi, 
             dx=dx, dy=dy)


########################################################################

# ENVT
landscape = surface_landscape
lmp.command(f"reset_timestep  0")
for step in range(0, NVT_2_steps, n_steps):
    for i in range(n_grid_x):
        for j in range(n_grid_y):
            surf = landscape[i, j]
            lmp.command(f"variable     z_center_{i*n_grid_y+j} equal {surf} ")
            lmp.command(f"variable     z_min_{i*n_grid_y+j}    equal {surf-3} ")
            lmp.command(f"variable     z_max_{i*n_grid_y+j}    equal {surf+3} ")
            lmp.command(f"region       surface_block_{i*n_grid_y+j} block {i * dx} {(i + 1) * dx} {j * dy} {(j + 1) * dy} INF INF")
            lmp.command(f"variable     k equal {k}")
            
            # AC E field
            # lmp.command(f"variable     f equal {Efreq}") # alternating E field
            # lmp.command(f"variable     pi equal 3.1415926") # alternating E field
            # lmp.command(f"variable     freq equal 'v_f*1e-15'") # alternating E field
            # lmp.command(f"variable     time equal 'step*0.1'") # alternating E field
            # lmp.command(f"variable     E equal '{E}*sin(2*v_pi*v_freq*v_time)'") # alternating E field
            
            # DC E field
            lmp.command(f"variable     E equal {E}")
            
            lmp.command(f"variable     efield_z_{i*n_grid_y+j}        atom '(v_E/(1.0+exp(-v_k*(z-v_z_center_{i*n_grid_y+j})))*(z>=v_z_min_{i*n_grid_y+j})*(z<=v_z_max_{i*n_grid_y+j}))'")
            lmp.command(f"variable     efield_up_{i*n_grid_y+j}       atom 'v_E * (z >= v_z_max_{i*n_grid_y+j})'")
            
            # lmp.command(f"variable     efield_up_{i*n_grid_y+j}       atom '{Ehigh} * (z >= v_z_max_{i*n_grid_y+j})'")
            # lmp.command(f"variable     efield_down_{i*n_grid_y+j}     atom '{Elow} * (z <= v_z_min_{i*n_grid_y+j})'")            
            lmp.command(f"fix          add_efield_{i*n_grid_y+j}      all efield 0.0 0.0 v_efield_z_{i*n_grid_y+j}    region surface_block_{i*n_grid_y+j}")
            lmp.command(f"fix          add_efield_up_{i*n_grid_y+j}   all efield 0.0 0.0 v_efield_up_{i*n_grid_y+j}   region surface_block_{i*n_grid_y+j}")
            # lmp.command(f"fix          add_efield_down_{i*n_grid_y+j} all efield 0.0 0.0 v_efield_down_{i*n_grid_y+j} region surface_block_{i*n_grid_y+j}")


    lmp.command("fix          integrate_E all rigid/nvt/small molecule temp 350.0 350.0 100.0")
    lmp.command("dump         3 all custom 100 nvtE.lammpstrj id type x y z")
    lmp.command("dump_modify  3 append yes")
    lmp.command(f"run          {n_steps}")
    lmp.command("unfix        integrate_E")
    lmp.command("undump       3")
    for i in range(n_grid_x):
        for j in range(n_grid_y):
            lmp.command(f"unfix          add_efield_{i*n_grid_y+j}")
            lmp.command(f"unfix          add_efield_up_{i*n_grid_y+j}")
            # lmp.command(f"unfix          add_efield_down_{i*n_grid_y+j}")
            lmp.command(f"region         surface_block_{i*n_grid_y+j} delete")
            lmp.command(f"variable       efield_z_{i*n_grid_y+j} delete")
            lmp.command(f"variable       efield_up_{i*n_grid_y+j} delete")
            # lmp.command(f"variable       efield_down_{i*n_grid_y+j} delete")
            
    coord = lmp.numpy.extract_atom('x')
    atom_type = lmp.numpy.extract_atom("type")
    box = lmp.extract_box()
    xlo, xhi = box[0][0], box[1][0]
    ylo, yhi = box[0][1], box[1][1]
    zlo, zhi = box[0][2], box[1][2]
    dx = (xhi - xlo) / n_grid_x
    dy = (yhi - ylo) / n_grid_y 
    surface_landscape = np.zeros((n_grid_x, n_grid_y))
    for i in range(n_grid_x):
        xrange = [i * dx, (i + 1) * dx]
        for j in range(n_grid_y):
            yrange = [j * dy, (j + 1) * dy]
            grid_mols = coord[(coord[:, 0] > xrange[0]) & 
                              (coord[:, 0] < xrange[1]) & 
                              (coord[:, 1] > yrange[0]) & 
                              (coord[:, 1] < yrange[1])]
            surface_landscape[i, j] = identify_surface(grid_mols, zlo, zhi)
    landscape = surface_landscape
