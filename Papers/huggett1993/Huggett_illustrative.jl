# Huggett exchange economy: readable value function iteration + Young transport.
# Run: julia Huggett_illustrative.jl
module HuggettIllustrative

using Interpolations: linear_interpolation
using Optim: optimize, minimizer
using QuantEcon: rouwenhorst, stationary_distributions
using LinearAlgebra, Printf

export parameters, utility, solve_household, policy_on_grid, lottery,
       stationary_distribution, solve_equilibrium

function parameters(; na=250, nd=1500, nz=7, beta=0.9, gamma=1.5,
                    rho=0.9, log_income_std=0.35, amin=-0.5, amax=20.0)
    # log income is AR(1). The standard deviation above is UNCONDITIONAL.
    chain = rouwenhorst(nz,rho,log_income_std*sqrt(1-rho^2))
    income_prob = stationary_distributions(chain)[1]
    income = exp.(chain.state_values)
    income ./= dot(income_prob,income)  # mean endowment = 1
    # Curved grids place more points near the borrowing constraint.
    a = amin .+ (amax-amin).*range(0,1;length=na).^2
    d = amin .+ (amax-amin).*range(0,1;length=nd).^2
    (; beta,gamma,rho,log_income_std,income,income_prob,P=chain.p,a,d)
end

utility(c,gamma) = c <= 0 ? -Inf : gamma == 1 ? log(c) : (c^(1-gamma)-1)/(1-gamma)

# Household: V(a,z) = max_a' u(a + e(z) - q*a') + beta E[V(a',z')|z].
function solve_household(q,p; initial=nothing,tol=1e-8,maxiter=10_000)
    V = isnothing(initial) ? zeros(length(p.a),length(p.income)) : copy(initial.V)
    policy = similar(V)
    for iteration in 1:maxiter
        EV = V*p.P'
        Vnew = similar(V)
        for z in eachindex(p.income)
            continuation = linear_interpolation(p.a,EV[:,z])
            for i in eachindex(p.a)
                resources = p.a[i]+p.income[z]
                lower,upper = first(p.a),min(last(p.a),(resources-1e-12)/q)
                upper > lower || error("Infeasible borrowing limit at q=$q")
                objective(ap) = -(utility(resources-q*ap,p.gamma)+p.beta*continuation(ap))
                solution = optimize(objective,lower,upper;abs_tol=1e-10)
                choices = (lower,minimizer(solution),upper)
                values = objective.(choices)
                best = argmin(values)
                policy[i,z],Vnew[i,z] = choices[best],-values[best]
            end
        end
        residual = maximum(abs.(Vnew-V))
        V = Vnew
        if residual < tol
            c = p.a .+ p.income' .- q.*policy
            return (;V,policy,c,iterations=iteration,residual)
        end
    end
    error("Value function iteration did not converge")
end

policy_on_grid(policy,p,grid=p.d) = hcat([
    linear_interpolation(p.a,policy[:,z]).(grid) for z in eachindex(p.income)]...)

# Young (2010): split an off-grid choice between its two neighboring nodes.
function lottery(grid,ap)
    first(grid)-1e-10 <= ap <= last(grid)+1e-10 || error("Policy outside asset grid")
    ap = clamp(ap,first(grid),last(grid))
    j = clamp(searchsortedlast(grid,ap),1,length(grid)-1)
    w = (ap-grid[j])/(grid[j+1]-grid[j])
    j,w
end

function stationary_distribution(policy,grid,P; initial=nothing,tol=1e-10,maxiter=100_000)
    mu = isnothing(initial) ? fill(1/length(policy),size(policy)) : copy(initial)
    for iteration in 1:maxiter
        next = zeros(size(mu))
        for z in axes(mu,2),i in axes(mu,1)
            j,w = lottery(grid,policy[i,z])
            for zp in axes(mu,2)
                # Saving depends on TODAY'S income z, for every tomorrow state zp.
                mass = mu[i,z]*P[z,zp]
                next[j,zp] += (1-w)*mass
                next[j+1,zp] += w*mass
            end
        end
        residual = sum(abs.(next-mu))
        mu = next
        residual < tol && return (;mu,iterations=iteration,residual)
    end
    error("Distribution iteration did not converge")
end

# Bonds are in zero net supply: choose their price so average assets equal zero.
function solve_equilibrium(p=parameters(); household_solver=solve_household,
        distribution_solver=stationary_distribution,warm_start=false,
        qlo=0.88,qhi=1.15,asset_tol=1e-5,maxiter=60,verbose=true)
    previous_household,previous_mu = nothing,nothing
    function at_price(q)
        household = household_solver(q,p;initial=warm_start ? previous_household : nothing)
        policy_d = policy_on_grid(household.policy,p)
        distribution = distribution_solver(policy_d,p.d,p.P;initial=warm_start ? previous_mu : nothing)
        previous_household,previous_mu = household,distribution.mu
        assets = sum(distribution.mu .* p.d)
        (;q,assets,household,policy_d,distribution)
    end
    low,high = at_price(qlo),at_price(qhi)
    low.assets > 0 && high.assets < 0 || error("Prices do not bracket zero asset demand")
    for iteration in 1:maxiter
        result = at_price((qlo+qhi)/2)
        verbose && @printf("%2d  q=%.8f  net assets=% .6g\n",iteration,result.q,result.assets)
        abs(result.assets) < asset_tol && return result
        if result.assets > 0
            qlo = result.q
        else
            qhi = result.q
        end
    end
    error("Bond market did not clear: inspect the grids and price bracket")
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    using .HuggettIllustrative
    @time result = solve_equilibrium()
    println("Borrowing-limit share: ",sum(result.distribution.mu[1,:]))
end
