# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

import numpy as np
from pyqula import geometry, embedding, potentials, algebra
import matplotlib.pyplot as plt
from scipy.linalg import inv

# Create a square lattice
g = geometry.square_lattice()  # 2D square lattice
h = g.get_hamiltonian()  # Generate Hamiltonian

# Create a defective Hamiltonian
hv = h.copy()  # Make a copy of the pristine Hamiltonian

# Add a single impurity at a specific site
impurity_site = 0  # Index of the site where you want to add the impurity
impurity_strength = 1.0  # Strength of the impurity potential
hv.add_onsite(potentials.impurity(g.r[impurity_site], v=impurity_strength))

# Create an embedding object
eb = embedding.Embedding(h, m=hv.intra)

# Parameters for calculations
delta = 0.01  # Broadening parameter
nk = 100  # Number of k-points for Brillouin zone integration

# Define a function to calculate the Green's function at arbitrary spatial points
def calculate_spatial_greens_function(embedding_obj, r1, r2, energy, delta, nk):
    """
    Calculate the Green's function G(r1, r2) between two arbitrary spatial points.
    
    In tight-binding, the Green's function at arbitrary points is:
    G(r1, r2) = sum_{i,j} phi_i(r1) G_{ij} phi_j*(r2)
    
    where:
    - phi_i(r) is the atomic orbital at site i evaluated at position r
    - G_{ij} is the Green's function matrix element between sites i and j
    
    Parameters:
    -----------
    embedding_obj : Embedding object
        The embedding object containing the system information
    r1, r2 : numpy arrays
        Spatial coordinates of the two points
    energy : float
        Energy at which to calculate the Green's function
    delta : float
        Broadening parameter
    nk : int
        Number of k-points for Brillouin zone integration
        
    Returns:
    --------
    G : complex
        The Green's function value G(r1, r2)
    """
    # Get the Green's function matrix at the atomic sites
    G_matrix = embedding_obj.get_gf(energy=energy, delta=delta, nk=nk)
    
    # Get the atomic positions
    atomic_positions = embedding_obj.H.geometry.r
    
    # Define the atomic orbital (using a simple exponential decay)
    def atomic_orbital(r, r_atom, decay=1.0):
        """Simple model for atomic orbital: exponential decay from atom center"""
        distance = np.linalg.norm(r - r_atom)
        return np.exp(-distance/decay)
    
    # Calculate G(r1, r2) using the tight-binding expansion
    G_r1_r2 = 0.0
    for i in range(len(atomic_positions)):
        for j in range(len(atomic_positions)):
            # Evaluate atomic orbitals at the given positions
            phi_i_r1 = atomic_orbital(r1, atomic_positions[i])
            phi_j_r2 = atomic_orbital(r2, atomic_positions[j])
            
            # Add contribution to the Green's function
            G_r1_r2 += phi_i_r1 * G_matrix[i, j] * phi_j_r2
    
    return G_r1_r2

# Function to calculate LDOS at an arbitrary spatial point
def calculate_ldos_at_point(embedding_obj, r, energy, delta, nk, decay=1.0):
    """
    Calculate LDOS at an arbitrary spatial point r.
    
    LDOS(r) = -1/π * Im[G(r, r)]
    
    Parameters:
    -----------
    embedding_obj : Embedding object
        The embedding object containing the system information
    r : numpy array
        Spatial coordinate
    energy : float
        Energy at which to calculate LDOS
    delta : float
        Broadening parameter
    nk : int
        Number of k-points for Brillouin zone integration
    decay : float
        Decay parameter for atomic orbitals
        
    Returns:
    --------
    ldos : float
        The LDOS value at point r
    """
    # Get the Green's function matrix at the atomic sites
    G_matrix = embedding_obj.get_gf(energy=energy, delta=delta, nk=nk)
    
    # Get the atomic positions
    atomic_positions = embedding_obj.H.geometry.r
    
    # Define the atomic orbital (using a simple exponential decay)
    def atomic_orbital(r, r_atom):
        """Simple model for atomic orbital: exponential decay from atom center"""
        distance = np.linalg.norm(r - r_atom)
        return np.exp(-distance/decay)
    
    # Calculate G(r, r) using the tight-binding expansion
    G_r_r = 0.0
    for i in range(len(atomic_positions)):
        for j in range(len(atomic_positions)):
            # Evaluate atomic orbitals at the given position
            phi_i_r = atomic_orbital(r, atomic_positions[i])
            phi_j_r = atomic_orbital(r, atomic_positions[j])
            
            # Add contribution to the Green's function
            G_r_r += phi_i_r * G_matrix[i, j] * phi_j_r
    
    # Calculate LDOS from the imaginary part of the Green's function
    ldos = -G_r_r.imag / np.pi
    
    return ldos

# Function to calculate LDOS on a dense spatial grid
def calculate_dense_spatial_ldos(embedding_obj, energy, delta, nk, grid_density=10, decay=1.0):
    """
    Calculate LDOS on a dense spatial grid.
    
    Parameters:
    -----------
    embedding_obj : Embedding object
        The embedding object containing the system information
    energy : float
        Energy at which to calculate LDOS
    delta : float
        Broadening parameter
    nk : int
        Number of k-points for Brillouin zone integration
    grid_density : int
        Number of grid points per unit length
    decay : float
        Decay parameter for atomic orbitals
        
    Returns:
    --------
    X, Y : 2D arrays
        Meshgrid of x and y coordinates
    ldos_grid : 2D array
        LDOS values on the grid
    """
    # Get the atomic positions to determine the grid bounds
    atomic_positions = embedding_obj.H.geometry.r
    
    # Determine the bounds of the system
    x_min, x_max = np.min(atomic_positions[:,0]), np.max(atomic_positions[:,0])
    y_min, y_max = np.min(atomic_positions[:,1]), np.max(atomic_positions[:,1])
    
    # Add some padding
    padding = 0.5
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding
    
    # Create a dense grid
    x_grid = np.linspace(x_min, x_max, int((x_max-x_min)*grid_density))
    y_grid = np.linspace(y_min, y_max, int((y_max-y_min)*grid_density))
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Get the Green's function matrix once (for efficiency)
    G_matrix = embedding_obj.get_gf(energy=energy, delta=delta, nk=nk)
    
    # Define the atomic orbital (using a simple exponential decay)
    def atomic_orbital(r, r_atom):
        """Simple model for atomic orbital: exponential decay from atom center"""
        distance = np.linalg.norm(r - r_atom)
        return np.exp(-distance/decay)
    
    # Calculate LDOS at each grid point
    ldos_grid = np.zeros_like(X)
    
    # Flatten the grid for easier iteration
    points = np.vstack((X.flatten(), Y.flatten())).T
    
    # Calculate LDOS for each point
    for idx, point in enumerate(points):
        # Convert to 2D indices
        i, j = np.unravel_index(idx, X.shape)
        
        # Calculate G(r, r) using the tight-binding expansion
        G_r_r = 0.0
        for m in range(len(atomic_positions)):
            for n in range(len(atomic_positions)):
                # Evaluate atomic orbitals at the given position
                phi_m_r = atomic_orbital(point, atomic_positions[m])
                phi_n_r = atomic_orbital(point, atomic_positions[n])
                
                # Add contribution to the Green's function
                G_r_r += phi_m_r * G_matrix[m, n] * phi_n_r
        
        # Calculate LDOS from the imaginary part of the Green's function
        ldos_grid[i, j] = -G_r_r.imag / np.pi
    
    return X, Y, ldos_grid

# 1. Calculate LDOS at atomic sites (standard method)
energy = 0.0
print(f"Calculating LDOS at energy {energy} at atomic sites")
x, y, ldos_values = eb.get_ldos(energy=energy, delta=delta, nk=nk)

# Plot the LDOS at atomic sites
plt.figure(figsize=(10, 8))
plt.scatter(x, y, c=ldos_values, cmap='hot', s=50)
plt.colorbar(label='LDOS')
plt.title(f'LDOS at E = {energy} (Atomic Sites Only)')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('ldos_atomic_sites.png')
plt.close()

# 2. Calculate LDOS on a dense spatial grid
print(f"Calculating LDOS at energy {energy} on a dense spatial grid")
grid_density = 10  # Points per unit length
decay = 1.0  # Decay parameter for atomic orbitals

X, Y, ldos_grid = calculate_dense_spatial_ldos(
    eb, 
    energy=energy, 
    delta=delta, 
    nk=nk, 
    grid_density=grid_density,
    decay=decay
)

# Plot the dense LDOS as a filled contour
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, ldos_grid, 100, cmap='hot')
plt.colorbar(label='LDOS')
plt.scatter(x, y, c='cyan', s=30, alpha=0.5, label='Atomic Sites')
plt.title(f'LDOS at E = {energy} (Dense Spatial Grid, decay={decay})')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('ldos_dense_spatial_grid.png')
plt.close()

# 3. Try different decay parameters to see the effect
decay_values = [0.5, 1.0, 2.0]
plt.figure(figsize=(15, 5))

for i, d in enumerate(decay_values):
    plt.subplot(1, 3, i+1)
    X, Y, ldos_grid = calculate_dense_spatial_ldos(
        eb, 
        energy=energy, 
        delta=delta, 
        nk=nk, 
        grid_density=grid_density,
        decay=d
    )
    plt.contourf(X, Y, ldos_grid, 100, cmap='hot')
    plt.colorbar(label='LDOS')
    plt.scatter(x, y, c='cyan', s=10, alpha=0.5)
    plt.title(f'decay = {d}')
    plt.xlabel('x')
    plt.ylabel('y')

plt.tight_layout()
plt.savefig('ldos_decay_comparison.png')
plt.close()

# 4. Calculate LDOS at different energies on a dense spatial grid
selected_energies = [-1.0, 0.0, 1.0]
plt.figure(figsize=(15, 5))

for i, e in enumerate(selected_energies):
    plt.subplot(1, 3, i+1)
    X, Y, ldos_grid = calculate_dense_spatial_ldos(
        eb, 
        energy=e, 
        delta=delta, 
        nk=nk, 
        grid_density=grid_density,
        decay=1.0
    )
    plt.contourf(X, Y, ldos_grid, 100, cmap='hot')
    plt.colorbar(label='LDOS')
    plt.title(f'LDOS at E = {e}')
    plt.xlabel('x')
    plt.ylabel('y')

plt.tight_layout()
plt.savefig('ldos_energy_comparison_dense.png')
plt.close()

# 5. Calculate DOS at multiple energies (standard calculation)
print("Calculating DOS at multiple energies")
energies = np.linspace(-4.0, 4.0, 101)
e_values, dos_defective = eb.multidos(energies=energies, delta=delta, nk=nk)

# Create an embedding object for the pristine system for comparison
eb_pristine = embedding.Embedding(h, m=h.intra)
e_values, dos_pristine = eb_pristine.multidos(energies=energies, delta=delta, nk=nk)

# Plot comparison of DOS
plt.figure(figsize=(10, 6))
plt.plot(e_values, dos_pristine, 'b-', label='Pristine', linewidth=2)
plt.plot(e_values, dos_defective, 'r-', label='With impurity', linewidth=2)
plt.xlabel('Energy')
plt.ylabel('DOS')
plt.title('Comparison of DOS: Pristine vs. With Impurity')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('dos_comparison.png')
plt.close()

print("All calculations completed. Check the output files and images.")
