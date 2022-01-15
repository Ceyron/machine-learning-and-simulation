![# Mathematics for Machine Learning and Simulation](https://user-images.githubusercontent.com/27728103/113576088-327ae700-961f-11eb-9611-05a9c8e7a0b1.png)

<a href="https://www.buymeacoffee.com/MLsim" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>

Here you can find all the material of my [YouTube Channel](https://www.youtube.com/channel/UCh0P7KwJhuQ4vrzc3IRuw4Q).

## Overview



Most of my videos are in English but some content is also offered in German. You can find the hand-written notes in the folders, repespectively.

These are the topics I cover at the moment:

* English:
    * **Math Basics** ([Playlist](https://www.youtube.com/watch?v=bYnT4wmXw4k&list=PLISXH-iEM4JnK1D8fkWMR2N0Sdw9QHfLb)): Things that are usually not taught in (engineering) math courses but that are relevant for Machine Learning & Simulation like (inequality) constrained optimization, some tricks in linear algebra, functionals, functional derivatives etc.
    * **Essential probability density/mass functions** ([Playlist](https://www.youtube.com/watch?v=dB2r4aLV_Ik&list=PLISXH-iEM4Jm5B_J9p1oUNGDAUeCFZLkJ)): Standard discrete probability mass functions like Bernoulli & Categorical as well as continuous proabability density functions like univariate and multivariate Gaussian/Normal together with their Maximum Likelihood Estimates, priors, posteriors, moments etc.
    * **Probabilistic Machine Learning** ([Playlist](https://www.youtube.com/watch?v=yBc01ZeaFxw&list=PLISXH-iEM4JlFsAp7trKCWyxeO3M70QyJ)): All the way from directed graphical models, the EM algorithm and Variational Inference to Deep Generative Models like Varitational Auto-Encoders, General Adverserial Networks and Latent Dirichlet Allocation
    * **Miscellaneous Computer Science Topics** ([Playlist](https://www.youtube.com/playlist?list=PLISXH-iEM4Jno71EGadIVpxVphfevVNM6)): Handy things that are relevant for some parts of Machine Learning and Simulation, like calling libraries in C from differen languages like Julia or Python.
    * **Sparse Matrices** ([Playlist](https://www.youtube.com/watch?v=BbbCVzJt1Xk&list=PLISXH-iEM4Jl8goS4m9UMzid0lUg64A9v)): Different ways to implement sparse matrices that become relevant when dealing with (large) sparse linear systems arising in simulation problems like FEM & CFD. All formats include an implementation in the C programming language.
    * **Continuum Mechanics** ([Playlist](https://www.youtube.com/watch?v=rxK-kQdH5qA&list=PLISXH-iEM4JlNGtteb5AvyIEyACp3wYIN)): The Fundamentals of Structural & Fluid Mechanics relevant for deriving numeric schemes in CFD & FEM. From Eulerian & Lagrangian description of motion to stretch & strain measures, to stress measures, time derivatives and constitutive modelling.
    * **Automatic Differentiation, Adjoints & Sensitivities** ([Playlist](https://www.youtube.com/watch?v=vlFN4qMtoH4&list=PLISXH-iEM4Jk27AmSvISooRRKH4WtlWKP)): Algrorithms and Mathematical Tricks to differentiate through various computer codes. These can include explicit computation graphs (like in Neural Networks), implicitly given relations like Linear or Nonlinear Systems or even Ordinary and Partial Differential equations. The equations are plenty, ranging from differentiable physics to classical Deep Learnig to Optimal Control. The derivations are accompanied by implementations in Python & Julia.
    * **Fenics Tutorial** ([Playlist](https://www.youtube.com/watch?v=QpA7E4YHbyU&list=PLISXH-iEM4Jl0-G1CpvG-mhrV0233tG_D)): A collection of videos to showcase the usage of the Fenics Finite Element Library to solve various Partial Differential Equations. Videos can be practical (including coding in Python) as well as theoretical on the Finite Element Method.
    * **Simulations simply implemented in Python or Julia** ([Playlist](https://www.youtube.com/watch?v=BQLvNLgMTQE&list=PLISXH-iEM4JmgBfU_QU262MQTYa7DoJK0)): My favorite series! If you ever wanted to write a Fluid Simulation from scratch, take a look at the playlist. Includes all kinds of simulations like CFD, Structural Mechanics, Electrodynamics etc.
* Deutsch:
    * **Tensor Analysis** ([Playlist](https://www.youtube.com/watch?v=x__XJjadiA8&list=PLISXH-iEM4JmfSEGOTDhEYfv0gXwqvX9B)): Grundlegende und erweiterte Techniken zur Mehrdimensionalen Analysis mit einen Fokus auf Visualierungen.
    * **Gewöhnliche Differentialgleichungen** ([Playlist](https://www.youtube.com/watch?v=DOWB8E8ji-A&list=PLISXH-iEM4Jlwa4FzRy_DdCQE4MO4dR0u)): Analytische und numerische Behandlung gewöhnlicher Differentialgleichungen, beginnend bei Trennung der Variablen und Variation der Konstanten bis hin zu Runge-Kutta Verfahren, Stabilitätsanalyse und Konvergenzuntersuchung.

These are topics I am going to cover in the long run:

* English:
    * Basics:
        * Tensor Calculus
        * Automatic Differentiation
        * More on Probability mass/density functions
    * Modelling & Simulation:
        * Ordinary Differential Equations (ODEs)
        * Partial Differential Equations (PDEs)
        * Linear Finite Element Method
        * (Numerical) Control Theory
        * Computational Fluid Dynamics
        * Nonlinear Finite Element Method
        * Visualization Techniques
        * Constitutive Modelling of Solids
        * Constitutive Modelling of Fluids
        * Computational Viscoelasticity
        * Compuational Plasticity
        * Uncertainty Quantification
    * Numerical Analysis:
        * Floating Point Error Analysis
        * Solving Linear Systems
        * Interpolation & Quadrature
        * Eigenvalue Computation
        * Solving Nonlinear Systems
        * Optimization Techniques
    * High-Performance Computing:
        * Essential topics of programming in parallel
        * A tour of the BLAS library
        * A tour of the lapack library
        * Parallel Numerics
        * PThread
        * OpenMP
        * MPI
        * CUDA
    * Machine Learning:
        * (Classical) Machine Learning
        * Dimensionality Reduction
        * Metrics in Machine Learning
        * Deep Learning
        * Markov-Chain Monte-Carlo Techniques

On top of that I have some ideas for projects. :) 

## Contribution

Contribution to this repo are always welcome. If you extended one of my source-codes for a more advanced example or if you think something is wrong or could have been explained better, feel free to open a Pull-Request to this repo. And of course if you can improve the code's performance (while maintaining readability), also feel free to open a pull request. 

## Donation

If you like the content of this repo, please consider ["buying me a coffee"](https://www.buymeacoffee.com/MLsim)
    
