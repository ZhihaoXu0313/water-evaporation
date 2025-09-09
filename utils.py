import numpy as np
import matplotlib.pyplot as plt


def plot_surface(filename, surface_landscape, xlo, xhi, ylo, yhi, dx, dy):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(np.arange(xlo, xhi, dx), np.arange(ylo, yhi, dy))
    ax.plot_surface(x, y, surface_landscape, cmap='viridis')
    plt.savefig(filename, dpi=300)

# def identify_surface(grid_mols, zlo, zhi, nbinsz=10):
#     dz = (zhi - zlo) / nbinsz # nbins = 10; dz = 75 / 10 = 7.5.
#     last = 0
#     surf = 0
#     for i in range(4, nbinsz):
#         zrange = [i * dz, (i + 1) * dz]
#         pinzbin = grid_mols[(grid_mols[:, 2] > zrange[0]) & (grid_mols[:, 2] < zrange[1])]
#         if len(pinzbin) == 0 and last == 0:
#             continue
#         if len(pinzbin) > 0.5 * last:
#             last = len(pinzbin)
#             surf = np.max(pinzbin[:, 2].reshape(-1, 1))
#         else:
#             return surf if len(pinzbin) < 0.2 * last else np.max(pinzbin[:, 2].reshape(-1, 1))
#     return 32.5

def identify_surface(grid_mols, zlo, zhi, nbinsz=50):
    z_bins = np.linspace(zlo, zhi, nbinsz + 1)
    z_vals = grid_mols[:, 2]
    counts, _ = np.histogram(z_vals, bins=z_bins)
    cumulative = np.cumsum(counts)
    cumulative = cumulative / np.max(cumulative)
    print(grid_mols, z_vals, counts)
    # Find where cumulative curve passes a threshold (e.g., 0.1 to 0.9 range)
    for i in range(1, len(cumulative)):
        if cumulative[i] > 0.9:
            return z_bins[i]
    return 32.5