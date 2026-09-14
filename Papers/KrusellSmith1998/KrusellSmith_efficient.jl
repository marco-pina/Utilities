# Same economy and outer KS loop; EGM and cached, in-place Young transport.
# Run: julia KrusellSmith_efficient.jl  (also writes krusell-smith.png)
module KrusellSmithEfficient

include("KrusellSmith_illustrative.jl")
using .KrusellSmithIllustrative
using Statistics, Printf
using Plots

export parameters, solve_household_egm, simulate_fast, solve, make_figure

# u'(c) = beta E[R' u'(c')]. The expectation is over BOTH future shocks;
# tomorrow's prices and consumption use the perceived K' for today's (K,z).
function solve_household_egm(B,p;initial=nothing,tol=1e-8,maxiter=10_000)
    resources = [prices(K,z,p).R*a+prices(K,z,p).income[e]
                 for a in p.a,e in 1:2,K in p.K,z in 1:2]
    c = isnothing(initial) ? copy(resources) : copy(initial.c)
    cnew,policy = similar(c),similar(c)
    a_endo = similar(p.a)
    na = length(p.a)
    for iteration in 1:maxiter
        for z in 1:2,k in eachindex(p.K)
            Kp = forecast(B,p.K[k],z)
            kp,wk = bracket(p.K,Kp)
            current = prices(p.K[k],z,p)
            Rp = (prices(Kp,1,p).R,prices(Kp,2,p).R)
            for e in 1:2
                for i in 1:na
                    expected = 0.0
                    for zp in 1:2,ep in 1:2
                        cp = (1-wk)*c[i,ep,kp,zp]+wk*c[i,ep,kp+1,zp]
                        expected += p.Pz[z,zp]*p.Pe[e,ep,z,zp]*Rp[zp]/cp
                    end
                    c_endo = 1/(p.beta*expected)
                    a_endo[i] = (c_endo+p.a[i]-current.income[e])/current.R
                end
                all(diff(a_endo).>0) || error("Endogenous grid is not increasing")
                for i in 1:na
                    if p.a[i]<=a_endo[1]
                        ap = 0.0
                    else
                        # Linear extrapolation only at the top of the endogenous
                        # asset grid, followed by the explicit asset ceiling.
                        j = clamp(searchsortedlast(a_endo,p.a[i]),1,na-1)
                        w = (p.a[i]-a_endo[j])/(a_endo[j+1]-a_endo[j])
                        ap = interpolate(p.a,j,w)
                    end
                    cash = resources[i,e,k,z]
                    policy[i,e,k,z] = clamp(ap,0.0,min(last(p.a),cash-1e-10))
                    cnew[i,e,k,z] = cash-policy[i,e,k,z]
                end
            end
        end
        residual = maximum(abs.(cnew-c))
        c,cnew = cnew,c
        residual<tol && return (;policy,c,iterations=iteration,residual)
    end
    error("Household EGM did not converge")
end

# Cache interpolation from the household grid to the distribution grid once.
# K and the policy change each period, so a fixed stationary sparse matrix is
# not reusable here. These small scatter loops avoid rebuilding one every time.
function simulate_fast(policy,p,zpath;initial=nothing)
    nd,T = length(p.d),length(zpath)-1
    T>p.burn || error("Simulation must extend beyond burn-in")
    dense_policy = zeros(nd,2,length(p.K),2)
    for z in 1:2,k in eachindex(p.K),e in 1:2,i in 1:nd
        j,w = bracket(p.a,p.d[i])
        dense_policy[i,e,k,z] = interpolate(@view(policy[:,e,k,z]),j,w)
    end
    mu = isnothing(initial) ? zeros(nd,2) : copy(initial)
    if isnothing(initial)
        j,w = bracket(p.d,mean(p.K))
        for e in 1:2
            share = e==1 ? p.u[zpath[1]] : 1-p.u[zpath[1]]
            mu[j,e],mu[j+1,e] = (1-w)*share,w*share
        end
    end
    next,average_mu = similar(mu),zeros(nd,2)
    K = zeros(T+1)
    mass_error = employment_error = upper_mass = 0.0
    for t in 1:T
        z,zp = zpath[t],zpath[t+1]
        capital = 0.0
        for e in 1:2,i in 1:nd
            capital += p.d[i]*mu[i,e]
        end
        K[t] = capital
        k,wk = bracket(p.K,capital)
        pr = prices(capital,z,p)
        fill!(next,0.0)
        t>p.burn && (average_mu .+= mu)
        for e in 1:2,i in 1:nd
            ap = (1-wk)*dense_policy[i,e,k,z]+wk*dense_policy[i,e,k+1,z]
            pr.R*p.d[i]+pr.income[e]-ap>0 || error("Nonpositive consumption")
            j,w = bracket(p.d,ap)
            for ep in 1:2
                mass = mu[i,e]*p.Pe[e,ep,z,zp]
                next[j,ep] += (1-w)*mass
                next[j+1,ep] += w*mass
            end
        end
        mu,next = next,mu
        mass_error = max(mass_error,abs(sum(mu)-1))
        employment_error = max(employment_error,abs(sum(@view mu[:,1])-p.u[zp]))
        upper_mass = max(upper_mass,sum(@view mu[end,:]))
    end
    K[end] = sum(mu.*p.d)
    bracket(p.K,K[end])
    (;K,z=zpath,mu,average_mu=average_mu/(T-p.burn),mass_error,employment_error,upper_mass)
end

solve(p=parameters(na=180,nd=1000,nK=9,periods=6000,burn=1000);kwargs...) =
    solve_equilibrium(p;household_solver=solve_household_egm,simulator=simulate_fast,kwargs...)

function make_figure(result,p;path=joinpath(@__DIR__,"krusell-smith.png"))
    ENV["GKSwstype"] = "100"
    # Fresh aggregate shocks: this figure is an out-of-sample check of the law.
    sim = simulate_fast(result.household.policy,p,aggregate_path(p;seed=p.seed+1))
    errors = forecast_errors(result.B,sim,p)
    mass = vec(sum(sim.average_mu;dims=2))
    edges = collect(range(0,last(p.d);length=101))
    bins = zeros(100)
    for (a,m) in zip(p.d,mass)
        bins[clamp(searchsortedlast(edges,a),1,100)] += m
    end
    upper = p.d[searchsortedfirst(cumsum(mass),0.9999)]
    default(fontfamily="sans-serif",dpi=180,guidefontsize=11,tickfontsize=10,
        titlefontsize=14,legendfontsize=10,gridalpha=0.15,
        left_margin=5Plots.mm,bottom_margin=5Plots.mm)
    distribution = bar((edges[1:end-1]+edges[2:end])/2,100bins;
        bar_width=diff(edges)[1],linewidth=0,color="#345b7d",label=false,
        xlabel="Assets",ylabel="Population per bin (%)",
        title="Time-averaged asset distribution",xlims=(0,upper))
    times = p.burn+1:min(p.burn+250,length(sim.K)-1)
    predicted = [forecast(result.B,sim.K[t],sim.z[t]) for t in times]
    capital = plot(1:length(times),sim.K[times.+1];label="Distribution-implied K'",
        color="#345b7d",linewidth=2,xlabel="Quarter after burn-in",ylabel="Aggregate capital",
        title=@sprintf("Unseen shocks | forecast RMSE %.4f%%",errors.rmse))
    plot!(capital,1:length(times),predicted;label="KS one-step forecast",color="#e4482f",
        linestyle=:dash,linewidth=2)
    figure = plot(distribution,capital;layout=(1,2),size=(1300,520),
        plot_title="Krusell-Smith | individual risk and aggregate uncertainty")
    savefig(figure,path)
    path
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    using .KrusellSmithEfficient
    p = parameters(na=180,nd=1000,nK=9,periods=6000,burn=1000)
    @time result = solve(p)
    println("Rows bad/good; columns intercept/slope:\n",result.B)
    println("One-step capital forecast RMSE (%): ",result.errors.rmse)
    println("Figure: ",make_figure(result,p))
end
