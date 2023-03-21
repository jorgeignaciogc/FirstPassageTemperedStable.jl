# FirstPassageTemperedStable.jl
This package provides a fast simulation algorithm for the first passage event of a tempered stable process across a target function. 


This package includes many auxiliary functions, including the following functions. (The notation used below follows that of the original article published by the same authors of this package.)

* The following function produces a sample of $S_t$ under $\mathbb{P}_0$ (i.e. a stable increment or marginal):
```julia
rand_stable(α::Real, θ::Real, t::Real)
```

* The following function produces a sample of $S_t$ under $\mathbb{P}_q$ (i.e. a tempered stable increment or marginal):
```julia
rand_tempered_stable(α::Real, θ::Real, q::Real, t::Real)
```

* The following function produces a sample of $S_t|\\{S_t\le s\\}$ under $\mathbb{P}_0$:
```julia
rand_small_stable(α::Real, θ::Real, t::Real, s::Real)
```

* The following function produces a sample of $S_{t}|\\{S_{t}\le s\\}$ under $\mathbb{P}_{q}$:
```julia
rand_small_tempered_stable(α::Real, θ::Real, q::Real, t::Real, s::Real)
```

* The following function produces a sample of the undershoot $S_{\tau-}|\\{S_{\tau-} < S_{\tau}\\}$ under $\mathbb{P}_{0}$ where $\tau$ is the crossing time and $s = b(\tau)$ is the crossing level 
```julia
rand_undershoot_stable(α::Real, θ::Real, t::Real, s::Real)
```

* The following function produces a sample of the vector $(\tau, S_{\tau-}, S_{\tau}-S_{\tau-})$ under $\mathbb{P}_0$ where $\tau$ is the crossing time across the boundary $b$, `b` is the boundary function $b$, `Db` is the derivative of `b` and `B` is the inverse function of $t \mapsto t^{-1/\alpha}b(t)$: 
```julia 
rand_crossing_stable(α::Real, θ::Real, b::Function, Db::Function, B::Function)
```

* The following function produces a sample of the vector $(\tau, S_{\tau-}, S_{\tau}-S_{\tau-})|\\{t\le T\\}$ under $\mathbb{P}_0$ where $\tau$ is the crossing time across the boundary $b$, `T` is some positive number, `b` is the boundary function $b$, `Db` is the derivative of `b` and `B` is the inverse function of $t \mapsto t^{-1/\alpha}b(t)$:
```julia
rand_crossing_small_stable(α::Real, θ::Real, b::Function, Db::Function, B::Function, T::Real)
```

* The following function produces a sample of the vector $(\tau, S_{\tau-}, S_{\tau}-S_{\tau-})$ under $\mathbb{P}_q$ where $\tau$ is the crossing time across the boundary $b$ and $b:t\mapsto \min\{a_0-a_1 t,r\}$ is the boundary function:
```julia
rand_crossing_tempered_stable(α::Real, θ::Real, q::Real, a0::Real, a1::Real, r::Real)
```

* The following function produces a sample of the vector $(\tau,Z_{\tau-},Z_{\tau}-Z_{\tau-})$ where $Z = Z^+-Z^-$ is the difference of two tempered stable subordinators, $\tau = \min\{\tau_0,T\}$ where $\tau_0$ is the crossing time $\tau_0=\inf\\{t>0:Z_t>r\\}$ across level `r`, $Z^+$ has parameters $(\alpha_1,\theta_1,q_1)$ and $Z^-$ has parameters $(\alpha_2,\theta_2,q_2)$:
```julia
rand_crossing_BV(α1::Real, θ1::Real, q1::Real, α2::Real, θ2::Real, q2::Real, T::Real, r::Real)
```

# Example: testing the marginals laws of the undershoot and overshoot

```julia
using Plots, SpecialFunctions, StatsBase, HypothesisTests, Random

α = .6
θ = 1.
b = 1.
t = (b/rand_stable(α,θ,1.))^α
gj_x, gj_w = gaussjacobi(2^10, -α, 0.)
FU(x) = CDF_undershoot(α, θ, t, b, x, gj_x, gj_w)

Random.seed!(2022)
n = 10000
@time XU = rand_undershoot_stable(α, θ, t, b)
@time XU = [rand_undershoot_stable(α, θ, t, b) for i = 1:n]
@time XT = [(b/rand_stable(α,θ,1.))^α for i = 1:n]
@time XU2 = [rand_undershoot_stable(α, θ, XT[i], b) for i = 1:n]
# 10k samples in 7.6857 seconds for α = .95
@time XO = (b .- XU2) .* (rand(n) .^ (-1/α) .- 1)

# Empirical CDFs
FO(x) = beta_inc(α, 1-α, b/(x+b))[2]
ECDF_U = ecdf(XU)
ECDF_O = ecdf(XO)
EFU(x) = ECDF_U(x)
EFO(x) = ECDF_O(x)

plot([EFU, FU], 0.01, b)
plot([EFO, FO], 0.01, 10)

# Rescaled to Uniform(0,1)

FO(x) = beta_inc(α, 1-α, b/(x+b))[2]
Id(x) = x
NXU = FU.(XU)
NXO = FO.(max.(XO,0))
NECDF_U = ecdf(NXU)
NECDF_O = ecdf(NXO)
NEFU(x) = NECDF_U(x)
NEFO(x) = NECDF_O(x)
plot([Id, NEFU, NEFO], 0.01, b)


# Statistical test:
ExactOneSampleKSTest(NXU, Uniform())
# Not rejected: p-value of 0.3216
ExactOneSampleKSTest(NXO, Uniform())
# Not rejected: p-value of 0.6090
```
