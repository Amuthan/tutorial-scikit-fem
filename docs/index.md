# An unofficial scikit-fem tutorial

**This tutorial is still under development. It is expected to be fully ready by September 2026.**

*Author:* Amuthan A. Ramabathiran  
*Email:* aramabat@calpoly.edu

Aerospace Engineering, Cal Poly (San Luis Obispo)

**No AI tools were used in creating the contents of this tutorial.** For students learning how to implement finite element method using this tutorial, the use of generative AI tools is *not recommended* until they gain sufficient mastery of the underlying theoretical and computational concepts.  

## About this tutorial

This tutorial provides a simple hands-on introduction to implementing the Finite Element Method (FEM) using the [scikit-fem](https://github.com/kinnala/scikit-fem) library. The [official documentation](https://scikit-fem.readthedocs.io/en/latest/index.html) has a good description of the library along with several examples. This tutorial is complementary to the official documentation and is designed specifically for teaching FEM implementation. Several details that are not explicitly stated in the official documentation are included here to ease the learning process. The tutorial introduces differential equations of increasing complexity, starting with one dimensional examples, and culminating in three dimensional problems.

## Why scikit-fem?

There are several excellent open-source FEM libraries like [Fenics](https://fenicsproject.org/), [Firedrake](https://www.firedrakeproject.org/), [NGSolve](https://ngsolve.org/), [Freefem](https://freefem.org/), [Deal.II](https://dealii.org/), and many more! Each of these have dedicated user groups, advantages and disadvantages. This tutorial focuses on the `scikit-fem` library for the following reasons:

- It is written entirely in Python and requires only `numpy` and `scipy` to install. This significantly reduces installation related issues in the classroom context. 
- It has a minimal and clean design that is useful from both the research and pedagogic viewpoints.
- It comes with a rich set of features, and is also relatively easy to extend when new features are needed.
- It is primarily a finite element assembly library. Any good linear solver library can be used to solve the resulting discretized equations.
- The source code is relatively small and easy to parse.
- The developers are very helpful when you run into issues.

In short, `scikit-fem` provides a simple, extensible, and clean framework to implement FEM for PDEs.

## Who is this for?

This tutorial was primarily developed to help students get started with programming simple FEM simulations. This is also likely to be useful to researchers looking for a good Python-based FEM library.

## Any and all feedback is welcome!

If you find this tutorial useful, or if you have any comments/suggestions for improvement, or if you want to share your experience using it to teach FEM, feel free to write to me at aramabat@calpoly.edu. 


