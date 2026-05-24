Experiment one in detail. Subcritical flow over a Gaussian bump, from Dazzi 2024.

Let me explain what this experiment is, how it works physically, how the PINN solves it, and what results to expect.

The physical setup.

Imagine a straight, flat channel, about 20 to 25 meters long. On the bottom of this channel, right in the middle, there is a smooth bump. The bump is parabolic, described by the formula: z b of x equals 0.2 minus 0.05 times x squared, but only between x equals minus 2 and x equals plus 2. Outside that range, the bottom is flat. So the bump is 4 meters wide at the base and 0.2 meters tall at the peak.

Water flows through this channel at a steady rate. The upstream boundary has a fixed discharge of 0.18 cubic meters per second per unit width. The downstream boundary has a fixed water depth. The Manning friction coefficient is 0.02, though some tests use zero friction. The flow is subcritical, meaning the Froude number is below one everywhere.

What happens physically.

When subcritical flow encounters a bottom bump, something counterintuitive happens. The water surface goes down over the bump, not up. This is because in subcritical flow, the surface elevation and the bed elevation have opposite signs in their spatial derivatives. The relationship is: d h over d x equals negative d z b over d x divided by one minus Froude number squared. So a bump on the bottom creates a depression on the surface. The velocity increases over the bump because the same discharge must pass through a shallower cross section, since Q equals u times h is constant.

This is actually very useful for inversion. If you can measure the surface depression, you can infer the bump underneath. The sensitivity depends on the Froude number. At low Froude numbers, typical of deep slow flows like estuaries, the surface response is weak, maybe only centimeters for a meter-scale bump. At higher Froude numbers, the surface response is stronger.

How Dazzi's PINN works.

The key innovation in Dazzi 2024 is the augmented shallow water equations. Normally, the SWE have three equations: continuity and two momentum equations, with h, u, and v as unknowns. Bathymetry z b is given as a known input. Dazzi adds a fourth equation: partial z b over partial t equals zero. This says the bed doesn't change in time. By adding this equation, z b becomes a learnable variable, and the PINN learns the bathymetry simultaneously with the flow field.

The neural network takes spatial coordinate x and time t as inputs, and outputs three variables: water depth h, velocity u, and bed elevation z b. The bed slopes, partial z b over partial x, are computed automatically via automatic differentiation of the network, which is consistent with how the PINN computes all other derivatives.

The network architecture is 7 hidden layers with 300 neurons each. The optimizer is Adam with exponential learning rate decay from ten to the minus three down to ten to the minus six, running for 30,000 epochs. Training takes about 7 to 8 minutes on an NVIDIA A100 GPU.

The loss function.

The total loss has five components. First, the PDE loss, which penalizes violations of all four augmented SWE equations at 1,000 randomly distributed collocation points. Second, the initial condition loss at 200 points, weighted by 100. Third, the boundary condition loss at 200 points, 100 per boundary, weighted by 10. Fourth, a depth positivity constraint to ensure h stays non-negative, weighted by 100. Fifth, the bed conservation constraint, which reinforces that z b should not change in time, weighted by 10.

The PDE loss weight is 1, so the initial condition loss is a hundred times stronger than the PDE loss. This aggressive weighting on initial and boundary conditions helps the PINN converge to the correct physical solution.

Results and accuracy.

For the steady flow over bump case, Dazzi reports a mean absolute error for depth h of about 5.1 times ten to the minus 3 meters, and an RMSE of about 1.1 times ten to the minus 2 meters. For velocity u, the MAE is about 1.4 times ten to the minus 3 meters per second. The bathymetry z b is recovered with errors on the order of ten to the minus 3 meters. The PINN captures the surface depression over the bump correctly, and the velocity increase is well reproduced.

However, Dazzi also found that the PINN struggles when the bottom has discontinuities, like a sharp step. This is the spectral bias problem: neural networks with smooth activation functions have difficulty representing sharp transitions. For our thesis, this is fine because real-world bathymetry in rivers and coasts is generally smooth.

How we would use this for bathymetry inversion.

Dazzi's paper frames this as a forward problem, meaning the network learns h, u, and z b simultaneously from the physics alone plus boundary conditions. For our thesis, we flip this into an explicit inverse problem. We observe the water surface elevation eta at some points, perhaps with noise, and we want to recover the unknown z b underneath.

The key modification is: instead of using initial conditions for z b, we remove knowledge of the bottom and let the PINN discover it from the surface observations plus the SWE physics. The loss function becomes: data loss on observed eta, plus SWE residual loss, plus regularization on z b such as total variation or Tikhonov smoothness penalties.

The sensitivity studies we would run on this case are: varying observation density from 100 percent of domain points down to 5 percent; adding Gaussian noise at 0, 1, 2, and 5 percent; using water level eta only versus velocity u only versus both combined; and testing with known versus unknown Manning coefficient, which creates the equifinality problem.

Why this case matters.

This is the simplest possible test of bathymetry inversion with PINNs. If the method cannot recover a smooth 1D Gaussian bump from steady-state water surface observations, it will never work on anything harder. It isolates the core question from all other complications: no time dependence, no wetting-drying, no 2D spatial patterns, no complex geometry. Every other experiment builds on this foundation.

Code is available on Zenodo from Dazzi 2024, and analytical solutions are available in the SWASHES library. Ruppenthal 2026 provides a non-machine-learning comparison using optimal control with finite elements on the same geometry.
