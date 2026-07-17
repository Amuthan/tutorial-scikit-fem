# Plane stress example
# scikit-fem tutorial
# 
# Amuthan Ramabathiran
# aramabat@calpoly.edu

import numpy as np
import matplotlib.pyplot as plt

import skfem
import skfem.visuals.matplotlib as skplot 
from skfem.helpers import dot 


@skfem.BilinearForm
def a_planestress(u, eta, ctx):
    E = ctx['E']
    nu = ctx['nu']
    eps_u = np.array([
        u.grad[0,0],
        u.grad[1,1],
        (u.grad[0,1] + u.grad[1,0])
    ])
    eps_eta = np.array([
        eta.grad[0,0],
        eta.grad[1,1],
        (eta.grad[0,1] + eta.grad[1,0])
    ])
    C = (E/(1 - nu**2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu)/2]
    ])
    return np.einsum('i...,ij,j...', eps_eta, C, eps_u)


@skfem.LinearForm
def l_planestress(eta, ctx):
    T = ctx['T'][0]
    return dot(T, eta)


if __name__ == '__main__':
    E = 7e10
    nu = 0.3

    T = np.array([0.0, 1e8])

    mesh = skfem.MeshTri().load('dogbone_hole.msh')
    
    elt_s = skfem.ElementTriP1()
    elt = skfem.ElementVector(elt_s)

    Vh_s = skfem.Basis(mesh, elt_s)
    Vh = skfem.Basis(mesh, elt)

    Vf = skfem.FacetBasis(mesh, elt, facets=mesh.boundaries['Top'])

    K = skfem.asm(
        a_planestress,
        Vh,
        E=E,
        nu=nu
    )
    F = skfem.asm(
        l_planestress,
        Vf,
        T=(T,)
    )

    dirichlet_dofs = Vh.get_dofs('Bottom')
    solver_data = skfem.condense(K, F, D=dirichlet_dofs)

    uh = skfem.solve(*solver_data, solver=skfem.solver_iter_krylov())

    skplot.plot(Vh, uh)
    skplot.show()

    idx_u, idx_v = Vh.split_indices()

    skplot.plot(Vh_s, uh[idx_u], shading='gouraud', colorbar=True, cmap='viridis')
    skplot.show()

    skplot.plot(Vh_s, uh[idx_v], shading='gouraud', colorbar=True, cmap='viridis')
    skplot.show()

    # Compute vertical displacement along horizontal axis
    x_probes = np.array([
        np.linspace(0.055, 0.1, 20),
        np.ones(20)*0.5
    ])
    disp_probes = Vh_s.probes(x_probes)
    v_probes = disp_probes @ uh[idx_v]

    plt.plot(x_probes[0], v_probes, 'bo-', lw=3)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('v')
    plt.show()

    # Stress calculation
    elt_0 = skfem.ElementTriP0()
    V0 = skfem.Basis(mesh, elt_0, quadrature=Vh.quadrature)

    uh_g = Vh.interpolate(uh)
    eps = np.array([
        uh_g.grad[0,0],
        uh_g.grad[1,1],
        (uh_g.grad[0,1] + uh_g.grad[1,0])
    ])
    C = (E/(1 - nu**2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu)/2]
    ])
    sig = np.einsum('ij,j...->i...', C, eps)

    sxx = V0.project(sig[0])
    syy = V0.project(sig[1])
    sxy = V0.project(sig[2])

    skplot.plot(V0, sxx, colorbar=True, cmap='viridis')
    skplot.show()

    skplot.plot(V0, syy, colorbar=True, cmap='viridis')
    skplot.show()

    skplot.plot(V0, sxy, colorbar=True, cmap='viridis')
    skplot.show()

    # Compute syy along horizontal axis 
    stress_probes = V0.probes(x_probes)
    syy_probes = stress_probes @ syy

    s_coeffs = np.polyfit(np.log(x_probes[0]), np.log(syy_probes), 1)
    print(f'Coefficents of power law fit (syy): {s_coeffs}')

    plt.plot(x_probes[0], np.exp(s_coeffs[1])*x_probes[0]**s_coeffs[0], 'b-', lw=2)
    plt.plot(x_probes[0], syy_probes, 'ko', markersize=4)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel(r'$\sigma_{yy}$')
    plt.show()

