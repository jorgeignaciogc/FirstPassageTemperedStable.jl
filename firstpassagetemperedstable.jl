using Distributions, StatsBase, SpecialFunctions, Random, FastGaussQuadrature

######################################
# Auxiliary parameters and functions #
######################################

# Number of nodes in Gaussian quadrature

n_nodes = 2^10

gl_x, gl_w = gausslegendre(n_nodes)

function σ(α::Real, x::Real)
    return x == 0. ? (1-α)*α^(α/(1-α)) : sin((1 - α)*pi*x) * sin(α*pi*x)^(α/(1-α)) * sin(pi*x)^(-1/(1-α))
end

function Dσ(α::Real, x::Real)   
    return x == 0. ? 0. : pi*σ(α,x)*(α^2*cot(α*pi*x) + (1-α)^2*cot((1-α)*pi*x) - cot(pi*x)) / (1-α)
end

function D2σ(α::Real, x::Real)   
    return pi^2*σ(α,x)*((α^2*cot(α*pi*x) + (1-α)^2*cot((1-α)*pi*x) - cot(pi*x))^2 / (1-α)^2 + (- α^3*csc(α*pi*x)^2 - (1-α)^3*csc((1-α)*pi*x)^2 + csc(pi*x)^2) / (1-α))
end

function DlogDσ(α::Real, x::Real)   
    aux = α^2*cot(α*pi*x) + (1-α)^2*cot((1-α)*pi*x) - cot(pi*x)
    return pi*aux/(1-α) + (- α^3*csc(α*pi*x)^2 - (1-α)^3*csc((1-α)*pi*x)^2 + csc(pi*x)^2) / aux
end

function Dlogσ(α::Real, x::Real)   
    return x == 0. ? 0. : pi*(α^2*cot(α*pi*x) + (1-α)^2*cot((1-α)*pi*x) - cot(pi*x)) / (1-α)
end

function D2logσ(α::Real, x::Real)   
    return pi^2 * (csc(pi*x)^2 - α^3*csc(α*pi*x)^2 - (1-α)^3*csc((1-α)*pi*x)^2) / (1-α)
end

function D2σ0(α::Real, x::Real)
    return pi^2*σ0(α,x)*(α^2*cot(α*pi*x) + (1-α)^2*cot((1-α)*pi*x) - cot(pi*x))^2 + pi^2*σ0(α,x)*(csc(pi*x)^2 - α^3*csc(α*pi*x)^2 - (1-α)^3*csc((1-α)*pi*x)^2)
end

function σ0(α::Real, x::Real)
    return sin((1-α)*pi*x)^(1-α) * sin(α*pi*x)^α / sin(pi*x)
end

function Dσ0(α::Real, x::Real)   
    return pi*σ0(α,x)*(α^2*cot(α*pi*x) + (1-α)^2*cot((1-α)*pi*x) - cot(pi*x))
end

function Dlogσ0(α::Real, x::Real)   
    return pi*(α^2*cot(α*pi*x) + (1-α)^2*cot((1-α)*pi*x) - cot(pi*x))
end

function D2σ0(α::Real, x::Real)
    return pi^2*σ0(α,x)*(α^2*cot(α*pi*x) + (1-α)^2*cot((1-α)*pi*x) - cot(pi*x))^2 + pi^2*σ0(α,x)*(csc(pi*x)^2 - α^3*csc(α*pi*x)^2 - (1-α)^3*csc((1-α)*pi*x)^2)
end

function Ψaux(a::Real, x::Real)
    ca = 1-a
    ax = a*x
    cax = ca*x
    gax = gamma(1+ax)
    gcax = gamma(1+cax)
    ecax = exp(cax)
    eax = exp(ax-1)
    axx = ax^ax
    caxx = cax^cax
    sqax = sqrt(2*pi*a*cax)

    C1 = x == 0 ? Inf : gax * eax * (a/ca+ax)^(1+cax) / ax^(x+1)
    C2 = gcax * ecax / caxx
    C3 = gax * eax * (1+1/cax)^(1+cax) / (axx * sqax)
    C4 = gcax * ecax / (caxx *sqax)

    Cmin = min(C1,C2,C3,C4)

    if C1 == Cmin
        return (1, exp(x)*gax*x^(a/ca)/(ca*C1))
    elseif C2 == Cmin
        return (2, exp(x)*gcax/C2)
    elseif C3 == Cmin 
        return (3, erf(sqax*sqrt(pi)/2) * exp(x) * gax * x^(a/ca) / (C3*sqax))
    else# C4 == Cmin
        return (4, erf(sqax*sqrt(pi)/2)*exp(x)*gcax/(C4*sqax))
    end
end

# Standardised PDF of S(1/θ)
function ϕ(α::Real, x::Real)
    aux = σ.(α, (1 .+ gl_x) ./ 2)
    return (α/(2-2*α)) * x^(-1/(1-α)) * sum(gl_w .* aux .* exp.(- aux .* x^(-α/(1-α))))
end

# PDF of S(t)
function g(α::Real, θ::Real, t::Real, x::Real)
    sc = (θ*t)^(-1/α)
    return ϕ(α, sc * x) * sc
end

# CDF of S(t)
function G(α::Real, θ::Real, t::Real, x::Real)
    sc = (θ*t)^(-1/α)
    aux = σ.(α, (1 .+ gl_x) ./ 2)
    return (1/2) * sum(gl_w .* exp.(- aux .* (sc * x)^(-α/(1-α))))
end

# CDF of the overshoot S(τ_b)-b for constant b>0
function CDF_overshoot(α::Real, b::Real, x::Real)
    return beta_inc(α, 1-α, b/(x+b))[2]
end

# CDF of the undershoot S(τ_b-)|τ_b=t for constant b>0
function CDF_undershoot(α::Real, θ::Real, t::Real, b::Real, x::Real)
    gj_x, gj_w = gaussjacobi(n_nodes, -α, 0.)
    return 1 - sum(gj_w .* g.(α,θ,t, (gj_x .+ 1) .* (b/2)) .* (gj_x .> (2*x/b-1))) * (2/b)^(α-1) * θ * t * α / (gamma(1-α) * g(α,θ,t,b))
end

function CDF_undershoot(α::Real, θ::Real, t::Real, b::Real, x::Real, gj_x::Array{<:Real}, gj_w::Array{<:Real})
    return return 1 - sum(gj_w .* g.(α, θ, t, (gj_x .+ 1) .* (b/2)) .* (gj_x .> (2*x/b-1))) * (2/b)^(α-1) * θ * t * α / (gamma(1-α) * g(α,θ,t,b))
end

# f = logconcave density on [0,1], with mode at 0 and f(0) = 1
# a = largest 2^{-i}, i≥1, such that f(a) > 1/4
# b = f(a) > 1/4 ≥ f(2*a) = c
function Devroye_logconcave(f::Function, a::Real, b::Real, c::Real, L::Real)
    if c != 0
        l1 = log(b)
        l2 = log(c)
        dl = l1-l2
        s = 1/(1 + b + c/dl)
        X = 0
        while true
            U = rand()
            if U < s
                X = rand() * a
            elseif U < (1+b)*s
                X = a * (1 + rand())
            else
                X = a * (2 + rand(Exponential()) / dl)
            end
            if X < L && rand() < f(X) / ( X < a ? 1. : (X < 2*a ? b : exp(((2*a-X)*l1 + (X-a)*l2)/a)))
                return X
            end
        end
    else
        l1 = log(b)
        s = 1/(1 + b)
        X = 0
        while true
            U = rand()
            if U < s
                X = rand() * a
            else
                X = a * (1 + rand())
            end
            if rand() < f(X) / ( X < a ? 1 : b )
                return X
            end
        end
    end
end

# Inverse of the σ function
function invσ(α,x)
    s0 = σ(α, 0.)
    if x <= s0
        return 0.
    end
    
    n = 6
    N = 60
    eps = 1e-15

    s = log(x) * (1-α)
    ς(x) = α*log(sin(α*pi*x)) + (1-α)*log(sin((1-α)*pi*x)) - log(sin(pi*x)) - s
    Dς(x) = pi*(α^2*cot(α*pi*x) + (1-α)^2*cot((1-α)*pi*x) - cot(pi*x))

    # Binary search
    M = 3/(pi*(1-α^3-(1-α)^3))
    y = 1
    ym = 0
    ςp, ςm = 1., ς(ym)

    i = 0
    while y - ym > eps && (y >= 1 || M * (y-ym) > ym*y^2*(1-y)) && i < N
        i += 1
        if y < 1
            mid1 = (y+ym)/2
            mid2 = (ym*ςp - y*ςm) / (ςp - ςm)
            ςmid1 = ς(mid1)
            ςmid2 = ς(mid2)
            if ςmid1 < 0
                if mid1 > mid2 || !isfinite(mid2)
                    ym, ςm = mid1, ςmid1
                elseif ςmid2 < 0
                    ym, ςm = mid2, ςmid2
                else
                    ym, ςm = mid1, ςmid1
                    y, ςp = mid2, ςmid2
                end
            else
                if mid1 < mid2 || !isfinite(mid2)
                    y, ςp = mid1, ςmid1
                elseif ςmid2 > 0 
                    y, ςp = mid2, ςmid2
                else
                    ym, ςm = mid2, ςmid2
                    y, ςp = mid1, ςmid1
                end
            end
        else
            mid = (y+ym)/2
            ςmid = ς(mid)
            if ςmid < 0
                ym, ςm = mid, ςmid
            else
                y, ςp = mid, ςmid
            end
        end
    end

    if y - ym > eps
    # Newton-Raphson
        for i = 1:n
            z = y - ς(y) / Dς(y) # Newton
            if z == y
                return z
            elseif z > 1 || z < 0
                z = y - ς(y)
                if z > 1
                    z = (y+1)/2
                elseif z < 0
                    z = y/2
                end
            end
            y = z
        end
    end

    return y
end

# Inverse of Σ: u ↦ uσ(u)^α on [0,z]
function invΣ(α,z,y)
    n = 6
    N = 100
    # Binary search
    function ς(x)
        return x*σ(α,x)^α - y
    end
    x0 = z
    ς0 = ς(z)
    x1 = 0.
    ς1 = -y

    i = 0
    M = 2*(σ(α,1/2)/σ(α,0))^α
    while true
        i += 1
        m1 = (x0+x1)/2
        ςm1 = ς(m1)
        m2 = (x0*ς1-x1*ς0)/(ς1-ς0)
        ςm2 = ς(m2)

        if ςm1 > 0
            if ςm2 > 0
                if m1 < m2
                    x0, ς0 = m1, ςm1
                else
                    x0, ς0 = m2, ςm2
                end
            else
                x0, ς0 = m1, ςm1
                x1, ς1 = m2, ςm2
            end
        else
            if ςm2 > 0
                x0, ς0 = m2, ςm2
                x1, ς1 = m1, ςm1
            else
                if m1 < m2
                    x1, ς1 = m2, ςm2
                else
                    x1, ς1 = m1, ςm1
                end
            end
        end

        aux1 = Dlogσ(α, x0)
        aux2 = D2logσ(α, x0)
        aux3 = Dlogσ(α, x1)
        if i > 10 && (i >= N || (x1-x0) * M * (1 + x0 * α * (aux2 + α * aux1^2)/2) < 1 + max(x1,0) * α * aux3)
            break
        end
    end

    x0 = (x1+x0)/2

    # Newton-Raphson
    for i = 1:n
        y0 = x0 - (x0 - y / σ(α,x0)^α) / (1 + x0*α*Dlogσ(α,x0))
        if y0 == x0
            return x0
        end
        x0 = y0
    end
    return x0
end

# Simulation from the PDF proportional to the positive function u ↦ (1-u)^(-r)*(a+b(1-u)) on [1/2,z]
function rand_neg_pow(r,a,b,z)
    if r != 2
        a2, b2 = a/(1-r), b/(2-r)
        C0 = a2 * 2^(r-1) + b2 * 2^(r-2)
        C = a2 * (1-z)^(1-r) + b2 * (1-z) - C0
        
        # Newton-Raphson
        x0 = z
        y = rand()*C + C0
        n = 100
        for i = 1:n
            aux = 1-x0
            y0 = x0 + (a2*aux^(1-r) + b2*aux^(2-r) - y)/(a*aux^(-r) + b*aux^(1-r)) 
            if y0 == x0
                return x0
            end
            x0 = y0
        end
        return x0
    else
        C0 = 2*a + b * log(2)
        C = a / (1-z) - b * log(1-z) - C0
        
        # Newton-Raphson
        x0 = z
        y = rand() * C + C0
        n = 100
        for i = 1:n
            aux = 1-x0
            y0 = x0 - (a/aux - b*log(aux) - y)/(a/aux^2 + b/aux) 
            if y0 == x0
                return x0
            end
            x0 = y0
        end
        return x0
    end
end

# Integral x ↦ ∫_0^x σ(u)^α*exp(-σ(u)s)du for s≥0 and x∈[0,1]
function int_tilt_exp_σ(α::Real, s::Real, x::Real)
    z = invσ(α,α/s)
    return int_tilt_exp_σ(α, s, x, z)
end

# If precomputed, z should be the critical point: z = invσ(α/s)
function int_tilt_exp_σ(α::Real, s::Real, x::Real, z::Real)
    if 0 < z < x
        # Break the integral in two: [0,z] and [z,x]
        σx1 = σ.(α, (gl_x .+ 1) .* (z/2))
        σx2 = σ.(α, (x+z)/2 .+ gl_x .* (x-z)/2)
        return sum(gl_w .* (σx1 .^ α .* exp.(- σx1 .* s))) * z/2 + sum(gl_w .* (σx2 .^ α .* exp.(- σx2 .* s))) * (x-z)/2
    else
        σx1 = σ.(α, (gl_x .+ 1) .* (x/2))
        return sum(gl_w .* (σx1 .^ α .* exp.(- σx1 .* s))) * x/2
    end
end

# If precomputed, z should be the critical point: z = invσ(α/s)
function alt_int_tilt_exp_σ(α::Real, s::Real, x::Real, z::Real)
    if 0 < z < x
        # Break the integral in two: [0,z] and [z,x]
        if x <= 1/2
            σx1 = σ.(α, (gl_x .+ 1) .* (z/2))
            σx2 = σ.(α, (x+z)/2 .+ gl_x .* (x-z)/2)
            return sum(gl_w .* (σx1 .^ α .* exp.(- σx1 .* s))) * z/2 + sum(gl_w .* (σx2 .^ α .* exp.(- σx2 .* s))) * (x-z)/2
        elseif z <= 1/2
            σx1 = σ.(α, (gl_x .+ 1) .* (z/2))
            I = sum(gl_w .* (σx1 .^ α .* exp.(- σx1 .* s))) * z/2 
            if z < 1/2          
                σx2 = σ.(α, (1/2+z)/2 .+ gl_x .* (1/2-z)/2)
                I += sum(gl_w .* (σx2 .^ α .* exp.(- σx2 .* s))) * (1/2-z)/2
            end
            y = gl_x .* (x/2-1/4) .+ (x/2+1/4)
            I += -(exp(-σ(α, x) * s) / Dσ0(α,x) - exp(-σ(α, 1/2) * s) / Dσ0(α,1/2)) * (1-α) / s
            return  I - sum(gl_w .* exp.(-σ.(α, y) .* s) .* D2σ0.(α, y) ./ Dσ0.(α, y) .^ 2) * (x/2-1/4) * (1-α) / s 
        else # 1/2 < z < x
            σx1 = σ.(α, (gl_x .+ 1) .* (1/4))
            I = sum(gl_w .* (σx1 .^ α .* exp.(- σx1 .* s))) * 1/4 
            y = gl_x .* (x/2-1/4) .+ (x/2+1/4)
            I += -(exp(-σ(α, x) * s) / Dσ0(α,x) - exp(-σ(α, 1/2) * s) / Dσ0(α,1/2)) * (1-α) / s
            return  I - sum(gl_w .* exp.(-σ.(α, y) .* s) .* D2σ0.(α, y) ./ Dσ0.(α, y) .^ 2) * (x/2-1/4) * (1-α) / s 
        end
    else
        if x <= 1/2
            σx1 = σ.(α, (gl_x .+ 1) .* (x/2))
            return sum(gl_w .* (σx1 .^ α .* exp.(- σx1 .* s))) * x/2
        else # 1/2 < x <= z
            σx1 = σ.(α, (gl_x .+ 1) .* (1/4))
            I = sum(gl_w .* (σx1 .^ α .* exp.(- σx1 .* s))) * 1/4 
            y = gl_x .* (x/2-1/4) .+ (x/2+1/4)
            I += -(exp(-σ(α, x) * s) / Dσ0(α,x) - exp(-σ(α, 1/2) * s) / Dσ0(α,1/2)) * (1-α) / s
            return  I - sum(gl_w .* exp.(-σ.(α, y) .* s) .* D2σ0.(α, y) ./ Dσ0.(α, y) .^ 2) * (x/2-1/4) * (1-α) / s 
        end
    end
end

# Integral x ↦ ∫_0^x exp(-σ(u)s)du for s≥0 and x∈[0,1]
function int_exp_σ(α::Real, s::Real, x::Real)
    y = σ.(α, (gl_x .+ 1) .* (x/2))
    return sum(gl_w .* exp.(- y .* s)) * x/2
end

# Integral x ↦ ∫_0^x σ(u)^α du for x∈[0,1]
function int_pow_σ(α::Real, x::Real)
    return sum(gl_w .* σ.(α, (gl_x .+ 1) .* (x/2)) .^ α) * x/2
end

# Simulation from the density proportional to
# u ↦ exp(-σ(u)s) on the interval [z,1] for some z∈(0,1)
function rand_exp_σ(α::Real, s::Real, z::Real)
    aux0 = σ(α,z)
    f(x) = exp(s * ( aux0 - σ(α, z+x)))
    aux = log(4) / s + aux0
    a1 = (1-z)/2
    while σ(α, z + a1) > aux
        a1 /= 2
    end
    a2 = f(a1)
    U = 0

    if a1 == (1-z)/2
        a4 = 1 / (1 + a2)
        while true
            V = rand(3)
            U = V[2] < a4 ? a1 * V[1] : a1 * (1+V[1])
            if V[3] < f(U) / (U < a1 ? 1 : a2)
                return U
            end
        end
    else
        a3 = f(2*a1)
        a4 = 1 / (1 + a2 + a3 / log(a2/a3))
        while true
            V = rand(3)
            U = V[2] < a4 ? a1 * V[1] : (V[2] < a4 * (1+a2) ? a1 * (1+V[1]) : a1 * (2 + log(1/V[1])/log(a2/a3)))
            if U < 1 
                if V[3] < f(U) / (U < a1 ? 1 : (U < 2*a1 ? a2 : exp(((2*a1-U)*log(a2) + (U-a1)*log(a3))/a1)))
                    return U
                end
            end
        end
    end
end

# Simulation from the density proportional to
# u ↦ exp(-σ(u)s) on the interval [0,1]
function rand_exp_σ(α::Real, s::Real)
    aux0 = σ(α,0)
    f(x) = 0 < x < 1 ? exp(s * ( aux0 - σ(α, x))) : 0.
    aux = log(4) / s + aux0
    a1 = 1/2
    while σ(α, a1) > aux
        a1 /= 2
    end
    a2 = f(a1)
    a3 = a1 == 1/2 ? 0. : f(2*a1)
    #println("Devroye for exp(-σ(u)s)")
    return Devroye_logconcave(f,a1,a2,a3,1.)
end

# Simulation from the density proportional to
# u ↦ σ(u)^α*exp(-σ(u)s) on [0,1]
function rand_tilt_exp_σ(α::Real, s::Real)
    r = α / (1-α)
    z = invσ(α, α/s)
    function f0(u)
        auxf = σ(α, u)
        return auxf ^ α * exp(auxf * (-s))
    end

    p = int_tilt_exp_σ(α, s, 1., z)

    # Initialise just in case (avoids certain julia errors)
    a = aux = 1.
    β = min(z,1/2)
    U = rand()

    if U > int_tilt_exp_σ(α, s, z, z) / p # Sample lies on [z,1], where f is log-concave
        #println("Devroye for σ(u)^α*exp(-σ(u)s)")
        a = (1-z)/2
        aux = 1/f0(z) # (s/α) ^ α * exp(α)
        g(x) = f0(z + x) * aux
        while g(a) <= 1/4
            a /= 2
        end
        return z + Devroye_logconcave(g, a, g(a), a == (1-z)/2 ? 0. : g(2*a), 1-z)
    elseif z <= 1/2 || U < int_tilt_exp_σ(α, s, β, z) / p # Sample lies on [0,β]. Here we sample from Σ: u ↦ uσ(u)^α
        #println("Sample from u ↦ u*σ(u)^α on [0,β]")
        C = β*σ(α,β)^α
        while true
            U = invΣ(α, β, rand()*C)
            if rand() < exp(-σ(α,U)*s) / (1 + U * α * Dlogσ(α,U))
                return U
            end
        end 
    elseif α < 1/2 # Sample lies on [1/2,z]
        #println("Sample from the density u ↦ (1-u)^(-r)*(a+b(1-u)) on [1/2,z]")
        a, b = sin(pi*(1-α)), pi*α*(1-α)*cos(pi*α)
        sc = 2^r * a^(1-α)
        while true
            U = rand_neg_pow(r,a,b,z)
            aux = σ(α,U)
            if rand() < aux^α * exp(-s*aux) * (1-U)^r * sc / (a + b*(1-U))
                return U
            end
        end
    elseif α == 1/2 # Sample lies on [1/2,z]
        #println("Sample from the density u ↦ (1-u)^(-1) on [1/2,z]")
        return 1 - exp(log(2*(1-z)) * rand())/2
    else # α>1/2, r>1, z>1/2 and sample lies on [1/2,z]
        #println("Sample from u ↦ ρ(u)^(r-1) on [1/2,z]")
        C0 = σ0(α,1/2)^(r-1)
        C = σ0(α,z)^(r-1) - C0
        c = Dlogσ0(α,1/2) / σ0(α,1/2)
        while true
            U = rand() * C + C0
            U = invσ(α, U^(1/(2*α-1)))
            if rand() < c*σ0(α,U)*exp(-s*σ(α,U))/Dlogσ0(α,U)
                return U
            end
        end
    end
end

# Auxiliary function that produces (b, Db, B)
# given the parameters of b : t ↦ min(a0 - a1*t, r)
function crossing_functions(α::Real, a0::Real, a1::Real, r::Real)
    if a0 < 0
        return (b, Db, B)
    elseif a1 == 0
        function b1(t) return min(a0,r) end
        function Db1(t) return 0 end
        function B1(t) return (min(a0,r)/t)^α end
        return (b1, Db1, B1)
    else
        function b2(t) return min(a0 - a1*t, r) end
        aux = (a0-r)/a1
        if aux > 0
            function Db2(t) return t > aux ? -a1 : 0 end
            function B2(t) 
                if t > r * aux^(-1/α) 
                    return (t/r)^(-α)
                else
                    ra = 1/α
                    x = (a0 - t * aux^ra) / a1
                    for i = 1:50
                        y = x - (t * x^ra + a1 * x - a0) / (a1 + ra * t * x^(ra-1))
                        if y == x
                            return x
                        else
                            x = y
                        end
                    end
                    return x
                end
            end
            return (b2, Db2, B2)
        else
            Db3(t) = -a1
            function B3(t) 
                ra = 1/α
                x = (t/a0 + (a1/a0)^ra)^(-α)
                for i = 1:50
                    y = x - (t * x^ra + a1 * x - a0) / (a1 + ra * t * x^(ra-1))
                    if y == x
                        return x
                    else
                        x = y
                    end
                end
                return x
            end
            return (b2, Db3, B3)
        end
    end
end

###########################
# Main simulation methods #
###########################

# Sample of S(t) under ℙ_0
function rand_stable(α::Real, θ::Real, t::Real)
    return (θ*t)^(1/α) * (σ(α, rand()) / rand(Exponential()))^((1-α)/α)
end

# Sample of S(t) under ℙ_q
function rand_tempered_stable(α::Real, θ::Real, q::Real, t::Real)
    λ = (θ*t)^(1/α)*q
    ξ = λ^α
    r = α/(1-α)
    (i,Cmin) = Ψaux(α,ξ)
    if i == 1
        ax = α*ξ
        aux = ξ^(r+1)
        while true
            U, V = rand(2)
            X = rand(Gamma(ax,1))
            ρ = σ0(α,U)^(r+1)
            X1 = X^(-r)
            if V <= ρ * X^(-ax) * X1 * exp(-ρ*aux*X1) * Cmin
                return X/q
            end
        end
    elseif i == 2
        cax = (1-α)*ξ
        aux = ξ^(r+1)
        while true
            U, V = rand(2)
            X = rand(Gamma(1+cax,1))
            s = σ0(α,U)^(1/α)*X^(-1/r)
            if V <= X^(-cax) * exp(-λ*s) * Cmin
                return (θ*t)^(1/α)*s
            end
        end
    elseif i == 3
        ax = α*ξ
        aux = ξ^(r+1)
        sc = pi*sqrt((1-α)*ax/2)
        while true
            U, V = rand(2)
            U = erfinv(U * erf(sc)) / sc
            X = rand(Gamma(ax,1))
            ρ = σ0(α,U)^(r+1)
            X1 = X^(-r)
            if V <= ρ * X^(-ax) * X1 * exp((1-α)*ax*U^2/2-ρ*aux*X1) * Cmin
                return X/q
            end
        end
    else# i == 4
        cax = (1-α)*ξ
        aux = ξ^(r+1)
        sc = pi*sqrt(α*cax/2)
        while true
            U, V = rand(2)
            U = erfinv(U * erf(sc)) / sc
            X = rand(Gamma(1+cax,1))
            s = σ0(α,U)^(1/α)*X^(-1/r)
            if V <= X^(-cax) * exp(α*cax*U^2/2-λ*s) * Cmin
                return (θ*t)^(1/α)*s
            end
        end
    end
end

# Sample of S(t)|{S(t)≤s} under ℙ_0
function rand_small_stable(α::Real, θ::Real, t::Real, s::Real)
    s1 = (θ * t / s^α)^(1/(1-α))
    aux0 = σ(α,0)
    f(x) = exp(s1 * ( aux0 - σ(α, x)))
    aux = log(4) / s1 + aux0
    a1 = 1/2
    while σ(α, a1) > aux
        a1 /= 2
    end
    a2 = f(a1)
    U = 0

    if a1 == 1/2
        a4 = 1 / (1 + a2)
        while true
            V = rand(3)
            U = V[2] < a4 ? a1 * V[1] : a1 * (1+V[1])
            if V[3] < f(U) / (U < a1 ? 1 : a2)
                break
            end
        end
    else
        a3 = f(2*a1)
        a4 = 1 / (1 + a2 + a3 / log(a2/a3))
        while true
            V = rand(3)
            U = V[2] < a4 ? a1 * V[1] : (V[2] < a4 * (1+a2) ? a1 * (1+V[1]) : a1 * (2 + log(1/V[1])/log(a2/a3)))
            if U < 1 
                if V[3] < f(U) / (U < a1 ? 1 : (U < 2*a1 ? a2 : exp(((2*a1-U)*log(a2) + (U-a1)*log(a3))/a1)))
                    break
                end
            end
        end
    end

    return (θ*t)^(1/α) * (rand(Exponential()) / σ(α, U) + s1)^(-(1-α)/α)
end

# Sample of S(t)|{S(t)≤s} under ℙ_q
function rand_small_tempered_stable(α::Real, θ::Real, q::Real, t::Real, s::Real)
    if 1/2 > max(exp(-q*s),exp(-q^α*θ*t)) # p > exp(q^α θ t-qs) = 𝔼_q[exp(q(S(t)-s))] ≥ ℙ_q[S(t)>s] ⟹ ℙ_q[S(t)<s] > 1-p
        while (x = rand_tempered_stable(α, θ, q, t)) > s end
        return x
    else # max(exp(-q*s),exp(-q^α*θ*t)) > 1/2, recall that p > exp(q^α θ t-qs) = 𝔼_q[exp(q(S(t)-s))] ≥ ℙ_q[S(t)>s] ⟹ ℙ_q[S(t)<s] > 1-p
        while (x = rand_small_stable(α, θ, t, s)) > -log(rand())/q end
        # Acceptance probability = ℙ_0[S(t)< E/q | S(t)<s] ≥ max(exp(-qs), exp(-q^α θ t)) ≥ 1/2
        return x
    end
end

# Sample of the undershoot S(t-)|{S(t-)<S(t)} under ℙ_0 where: 
# t is the crossing time and s = b(t) is the crossing level
function rand_undershoot_stable(α::Real, θ::Real, t::Real, s::Real)
    r = α / (1-α)
    sc = (θ * t)^(1/α)
    s1 = s / sc
    s2 = s1^(-r)
    z = invσ(α,α/s2)
    p = (2-2^α)^(-α) * r^(-α) * s1^(α*r) *  int_exp_σ(α, s2, 1.) / (gamma(1-α) * int_tilt_exp_σ(α, s2, 1., z))
    p = 1/(1+1/p)
    c1 = (1-2^(α-1))^(-α) * s1^(-α)
    c2 = (2*r)^α * s2
    
    E = 0.

    while true
        if rand() <= p
            U = rand_exp_σ(α, s2)
            E = rand(Exponential()) / σ(α,U) + s2
        else
            U = rand_tilt_exp_σ(α, s2)
            E = rand(Gamma(1-α,1)) / σ(α,U) + s2
        end
        if rand() < abs(s1 - E^(-1/r))^(-α) / (c1 + c2 * ( E > s2 ? (E - s2)^(-α) : 0.))
            return sc * E^(-1/r)
        end 
    end    
end

# Sample of the vector (t, S(t-), S(t)-S(t-)) under ℙ_0 where: 
# t is the crossing time across the boundary b
# b : ℝ → ℝ is the target/boundary function
# Db : ℝ → ℝ is the derivative of b
# B : ℝ → ℝ  is the inverse function of t ↦ t^(-1/α)b(t)  
function rand_crossing_stable(α::Real, θ::Real, b::Function, Db::Function, B::Function)
    S = rand_stable(α, θ, 1)
    T = B(S)
    w0 = -Db(T)
    if w0 != 0
        w1 = b(T)/(α*T)
        if rand() <= w0 / (w0 + w1)
            return (T, b(T), 0.)
        end
    end
    U = rand_undershoot_stable(α, θ, T, b(T))
    return (T, U, (b(T)-U) * rand()^(-1/α))
end

# Sample of the vector (t, S(t-), S(t)-S(t-))|{t≤T} under ℙ_0 where: 
# T>0 is some time horizon
# t is the crossing time across the boundary b
# b : ℝ → ℝ is the target/boundary function
# Db : ℝ → ℝ is the derivative of b
# B : ℝ → ℝ is the inverse function of t ↦ t^(-1/α)b(t)  
function rand_crossing_small_stable(α::Real, θ::Real, b::Function, Db::Function, B::Function, T::Real)
    S = rand_stable(α, θ, 1)
    aux = T^(-1/α)*b(T)
    while S < aux
        S = rand_stable(α, θ, 1)
    end
    t = B(S)
    w0 = -Db(t)
    if w0 != 0
        w1 = b(t)/(α*t)
        if rand() <= w0 / (w0 + w1)
            return (t, b(t), 0.)
        end
    end
    U = rand_undershoot_stable(α, θ, t, b(t))
    return (t, U, (b(t)-U) * rand()^(-1/α))
end

# Sample of the vector (t, S(t-), S(t)-S(t-)) under ℙ_q where: 
# t is the crossing time across the boundary b
# b is the target/boundary function
# Db is the derivative of b
# BB : ℝ^2 → {f : ℝ → ℝ} so that B = BB(x0,x1) : ℝ → ℝ is the inverse function of t ↦ t^(-1/α)(b(t+x0)-x1)  
function rand_crossing_tempered_stable(α::Real, θ::Real, q::Real, b::Function, Db::Function, BB::Function)
    Tmax = (2*q*b(0)+1-2^(-α))/((2^α-1)*q^α*θ)
    Tf = Uf = 0
    while (S = rand_tempered_stable(α,θ,q,Tmax)) < b(Tmax+Tf) - Uf
        Tf += Tmax
        Uf += S
    end
    locb(x) = b(x+Tf) - Uf
    locDb(x) = Db(x+Tf)
    locB = BB(Tf,Uf)

    (T,U,V) = rand_crossing_small_stable(α,θ,locb,locDb,locB,Tmax)
    S = rand_stable(α,θ,Tmax-T)
    while q*(S+U+V) > -log(rand()) 
        (T, U, V) = rand_crossing_small_stable(α,θ,locb,locDb,locB,Tmax)
        S = rand_stable(α,θ,Tmax-T)
    end
    return (Tf+T, Uf+U, V)
end

# Sample of the vector (t, S(t-), S(t)-S(t-)) under ℙ_q where: 
# t is the crossing time across the boundary b
# b : t ↦ min(a0 - a1*t, r) is the target/boundary function
function rand_crossing_tempered_stable(α::Real, θ::Real, q::Real, a0::Real, a1::Real, r::Real)
    if q == 0 return rand_crossing_stable(α, θ, a0, a1, r) end

    R0 = (2^α - 1) / (2 * q)
    Tf = Uf = V = 0
    function locb(t) return 0 end
    function locDb(t) return 0 end
    function locB(t) return 0 end

    while Uf + V <= min(a0 - a1*Tf, r)
        Uf += V
        R = R0 + Uf
        Tmax = (2*q*min(a0,r,R-Uf)+1-2^(-α))/((2^α-1)*q^α*θ)
        while (S = rand_tempered_stable(α,θ,q,Tmax)) < min(a0 - a1*(Tmax+Tf), r, R) - Uf
            Tf += Tmax
            Uf += S
        end

        (locb,locDb,locB) = crossing_functions(α, a0-a1*Tf, a1, min(r,R) - Uf)
        (T,U,V) = rand_crossing_small_stable(α,θ,locb,locDb,locB,Tmax)
        S = rand_stable(α,θ,Tmax-T)

        while q*(S+U+V) > locb(Tmax) - log(rand()) 
            (T,U,V) = rand_crossing_small_stable(α,θ,locb,locDb,locB,Tmax)
            S = rand_stable(α,θ,Tmax-T)
        end
        Tf += T
        Uf += U
    end
    return (Tf, Uf, V)
end

# Sample of (τ,Z(τ-),Z(τ)-Z(τ-)) where Z = Z^+-Z^-
# At the stopped time τ = τ0∧T where τ0 is the crossing time τ0=inf{t>0:Z_t>r} across level r
# Z^+ has parameters (α1,θ1,q1) and Z^- has parameters (α2,θ2,q2)
function rand_crossing_BV(α1::Real, θ1::Real, q1::Real, α2::Real, θ2::Real, q2::Real, T::Real, r::Real)
    tt = H = 0
    b = r
    (t,u,v) = rand_crossing_tempered_stable(α1,θ1,q1,b,0,b)
    w = rand_tempered_stable(α2,θ2,q2,t)
    while u + v - w < b
        if tt + t >= T
            v = rand_small_tempered_stable(α1,θ1,q1,T-tt,b) - rand_tempered_stable(α2,θ2,q2,T-tt)
            return (T,H+v,0)
        end
        tt += t
        H += v - w
        b -= v - w
        (t,u,v) = rand_crossing_tempered_stable(α1,θ1,q1,b,0,b)
        w = rand_tempered_stable(α2,θ2,q2,t)
    end
    if tt + t >= T
        v = rand_small_tempered_stable(α1,θ1,q1,T-tt,b) - rand_tempered_stable(α2,θ2,q2,T-tt)
        return (T,H+v,0)
    end
    return (tt+t,H+u-w,v)
end