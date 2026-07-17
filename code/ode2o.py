# Solve second order ODE
# -u''(x) = 4c pi^2 sin (2 pi x), x \in (0,1)
# u(0) = 0, u(1) = 1
#
# Amuthan Ramabathiran
# Skfem tutorial


import numpy as np
import skfem 
from skfem.helpers import dot, grad
import skfem.visuals.matplotlib as skplot 


@skfem.LinearForm
def l(v, ctx):
    x = ctx['x']
    f = 4 * np.pi**2 * np.sin(2 * np.pi * x[0])
    return f * v 

@skfem.BilinearForm
def a(u, v, ctx):
    return dot( grad(u), grad(v) )


def u_exact(x):
    return np.sin(2 * np.pi * x)

@skfem.Functional
def sq_error_L2(ctx):
    u = ctx['u']
    x = ctx['x']
    return (u - u_exact(x[0]))**2 


if __name__ == '__main__':
    mesh = skfem.MeshLine( np.linspace(0, 1, 11) )
    skplot.draw(mesh)
    skplot.show()

    elt = skfem.ElementLineP1()

    Vh = skfem.Basis(mesh, elt)

    K = skfem.asm(a, Vh)
    F = skfem.asm(l, Vh) 

    dirichlet_dofs = Vh.get_dofs()

    solver_data = skfem.condense(K, F, D=dirichlet_dofs)
    uh = skfem.solve(*solver_data)

    skplot.plot(Vh, uh)
    skplot.show()

    uh_g = Vh.interpolate(uh)
    L2_error = np.sqrt(
        skfem.asm(sq_error_L2, Vh, u=uh_g)
    )

    print(f'FE L2 error = {L2_error:.3e}')