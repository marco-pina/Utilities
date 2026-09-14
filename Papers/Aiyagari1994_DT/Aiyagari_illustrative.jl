# Aiyagari production economy: readable VFI + Young distribution + market clearing.
# Run: julia Aiyagari_illustrative.jl
module AiyagariIllustrative

using Interpolations: linear_interpolation
using Optim: optimize, minimizer
using LinearAlgebra, Printf

export parameters, utility, firm_prices, solve_household, policy_on_grid,
       lottery, stationary_distribution, solve_equilibrium

function parameters(;na=250,nd=1500,beta=0.96,gamma=1.0,
                    alpha=0.33,delta=0.05,technology=1.0,amax=30.0)
    # Preserve the original two-state productivity process.
    income = [0.1,1.0]
    P = [0.95 0.05;0.1 0.9]
    income_prob = [P[2,1],P[1,2]]/(P[1,2]+P[2,1])
    labor = dot(income_prob,income)  # 0.4 efficiency units, NOT 1
    a = amax.*range(0,1;length=na).^2
    d = amax.*range(0,1;length=nd).^2
    (;beta,gamma,alpha,delta,technology,income,income_prob,P,labor,a,d)
end

utility(c,gamma) = c <= 0 ? -Inf : gamma == 1 ? log(c) : (c^(1-gamma)-1)/(1-gamma)

# Firms: Y=A*K^alpha*L^(1-alpha), r=MPK-delta, w=MPL.
function firm_prices(r,p)
    capital_labor_ratio = (p.alpha*p.technology/(r+p.delta))^(1/(1-p.alpha))
    w = (1-p.alpha)*p.technology*capital_labor_ratio^p.alpha
    demand = capital_labor_ratio*p.labor
    (;w,demand)
end

# Household budget: c = w*z + (1+r)*a - a'. Borrowing is not allowed.
function solve_household(r,w,p;initial=nothing,tol=1e-8,maxiter=10_000)
    V = isnothing(initial) ? zeros(length(p.a),length(p.income)) : copy(initial.V)
    policy = similar(V)
    for iteration in 1:maxiter
        EV = V*p.P'
        Vnew = similar(V)
        for z in eachindex(p.income)
            continuation = linear_interpolation(p.a,EV[:,z])
            for i in eachindex(p.a)
                resources = (1+r)*p.a[i]+w*p.income[z]
                lower,upper = first(p.a),min(last(p.a),resources-1e-12)
                upper > lower || error("No feasible positive consumption")
                objective(ap) = -(utility(resources-ap,p.gamma)+p.beta*continuation(ap))
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
            c = (1+r).*p.a .+ w.*p.income' .- policy
            return (;V,policy,c,iterations=iteration,residual)
        end
    end
    error("Value function iteration did not converge")
end

policy_on_grid(policy,p,grid=p.d) = hcat([
    linear_interpolation(p.a,policy[:,z]).(grid) for z in eachindex(p.income)]...)

function lottery(grid,ap)
    first(grid)-1e-10 <= ap <= last(grid)+1e-10 || error("Policy outside asset grid")
    ap = clamp(ap,first(grid),last(grid))
    j = clamp(searchsortedlast(grid,ap),1,length(grid)-1)
    w = (ap-grid[j])/(grid[j+1]-grid[j])
    j,w
end

# Distribute mass using TODAY'S productivity policy, then tomorrow's income risk.
function stationary_distribution(policy,grid,P;initial=nothing,tol=1e-10,maxiter=100_000)
    mu = isnothing(initial) ? fill(1/length(policy),size(policy)) : copy(initial)
    for iteration in 1:maxiter
        next = zeros(size(mu))
        for z in axes(mu,2),i in axes(mu,1)
            j,w = lottery(grid,policy[i,z])
            for zp in axes(mu,2)
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

# Stationary equilibrium: household assets = firms' demand for capital.
function solve_equilibrium(p=parameters();household_solver=solve_household,
        distribution_solver=stationary_distribution,warm_start=false,
        rlo=0.005,rhi=0.04,asset_tol=1e-5,maxiter=60,verbose=true)
    previous_household,previous_mu = nothing,nothing
    function at_rate(r)
        prices = firm_prices(r,p)
        household = household_solver(r,prices.w,p;initial=warm_start ? previous_household : nothing)
        policy_d = policy_on_grid(household.policy,p)
        distribution = distribution_solver(policy_d,p.d,p.P;initial=warm_start ? previous_mu : nothing)
        previous_household,previous_mu = household,distribution.mu
        capital = sum(distribution.mu .* p.d)
        excess = capital-prices.demand
        (;r,w=prices.w,capital,demand=prices.demand,excess,household,policy_d,distribution)
    end
    low,high = at_rate(rlo),at_rate(rhi)
    low.excess < 0 && high.excess > 0 || error("Interest rates do not bracket market clearing")
    for iteration in 1:maxiter
        result = at_rate((rlo+rhi)/2)
        verbose && @printf("%2d  r=%.8f  capital gap=% .6g\n",iteration,result.r,result.excess)
        abs(result.excess) < asset_tol && return result
        if result.excess > 0
            rhi = result.r
        else
            rlo = result.r
        end
    end
    error("Capital market did not clear: inspect the grids and interest-rate bracket")
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    using .AiyagariIllustrative
    @time result = solve_equilibrium()
    println("Capital: ",result.capital," | wage: ",result.w)
end
