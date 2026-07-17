# Heat transfer 2D
# scikit-fem tutorial
#
# Amuthan Ramabathiran
# aramabat@calpoly.edu

import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

import skfem
import skfem.visuals.matplotlib as skplot
from skfem.helpers import dot, grad


@skfem.BilinearForm
def a_heat_1(T, eta, ctx):
    kappa = ctx['kappa']
    return kappa * dot(grad(T), grad(eta))

@skfem.BilinearForm
def a_heat_2(T, eta, ctx):
    H = ctx['H']
    L = ctx['L']
    W = ctx['W']
    lC = L - 2*W
    return H*lC*T*eta


def q(x, d, R, Q):
    return ((x[0] - d)**2 + (x[1] - d)**2 - R**2)*Q

@skfem.LinearForm
def l_heat_1(eta, ctx):
    d = ctx['d']
    R = ctx['R']
    Q = ctx['Q']
    x = ctx['x']
    return q(x, d, R, Q) * eta

@skfem.LinearForm
def l_heat_2(eta, ctx):
    H = ctx['H']
    L = ctx['L']
    W = ctx['W']
    lC = L - 2*W
    T1 = ctx['T1']
    return H*lC*T1*eta


def convective_boundary(x):
    return reduce(
        np.logical_and,
        (
            np.isclose(x[0], L),
            x[1] > W,
            x[1] < L - W
        )
    )


if __name__ == '__main__':
    kappa = 1.0
    H = 5.0
    d = 0.75
    R = 0.5
    T0 = 10.0
    T1 = 1.0
    Q = 10.0

    L = 2.5
    W = 0.5

    Nx = 40
    Ny = 40

    mesh = skfem.MeshTri().init_tensor(
        np.linspace(0, L, Nx+1),
        np.linspace(0, L, Ny+1)
    )

    skplot.draw(mesh)
    skplot.show()

    mesh = mesh.with_boundaries({
        'Temperature': lambda x: np.isclose(x[1], L),
        'Convective': convective_boundary
    })

    elt = skfem.ElementTriP1()

    Vh = skfem.Basis(mesh, elt)
    Vf = skfem.FacetBasis(mesh, elt, facets=mesh.boundaries['Convective'])

    K = (
        skfem.asm(
            a_heat_1,
            Vh,
            kappa=kappa
        )
        +
        skfem.asm(
            a_heat_2,
            Vf,
            H=H,
            L=L,
            W=W
        )
    )
    F = (
        skfem.asm(
            l_heat_1,
            Vh,
            d=d,
            R=R,
            Q=Q
        )
        +
        skfem.asm(
            l_heat_2,
            Vf,
            H=H,
            L=L,
            W=W,
            T1=T1
        )
    )

    dirichlet_dofs = Vh.get_dofs('Temperature')
    T_dirichlet = Vh.zeros()
    T_dirichlet[dirichlet_dofs.all()] = T0

    solver_data = skfem.condense(K, F, x=T_dirichlet, D=dirichlet_dofs)

    Th = skfem.solve(*solver_data)

    skplot.plot(Vh, Th, shading='gouraud', colorbar=True, cmap='viridis')
    skplot.show()

    x_probes = np.array([
        [d, L-d, L-d, d, L/2],
        [d, d, L-d, L-d, L/2]
    ])
    Vh_probes = Vh.probes(x_probes)
    T_probes = Vh_probes @ Th

    for i_probe in range(x_probes.shape[1]):
        print(f'Temperature at ({x_probes[0,i_probe]}, {x_probes[1,i_probe]}) = {T_probes[i_probe]:.2f}')
