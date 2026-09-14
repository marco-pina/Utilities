# Same Aiyagari economy; EGM households, sparse Young transport, and warm starts.
# Run: julia Aiyagari_efficient.jl  (also writes aiyagari.png)
module AiyagariEfficient

include("Aiyagari_illustrative.jl")
using .AiyagariIllustrative
using Interpolations: linear_interpolation, Line
using LinearAlgebra, SparseArrays, Printf
using Plots

export parameters, solve_household_egm, stationary_distribution_fast, solve, make_figure

# Euler equation: c^(-gamma) = beta*(1+r)*E[c_next^(-gamma)].
# Fix a' first, recover c and today's endogenous assets, then interpolate back.
function solve_household_egm(r,w,p;initial=nothing,tol=1e-9,maxiter=20_000)
    c = isnothing(initial) ? (1+r).*p.a .+ w.*p.income' .- first(p.a) : copy(initial.c)
    policy,cnew = similar(c),similar(c)
    for iteration in 1:maxiter
        expected = c.^(-p.gamma)*p.P'
        c_endo = (p.beta*(1+r).*expected).^(-1/p.gamma)
        for z in eachindex(p.income)
            a_endo = (c_endo[:,z] .+ p.a .- w*p.income[z])./(1+r)
            all(diff(a_endo).>0) || error("Endogenous grid is not increasing")
            inverse = linear_interpolation(a_endo,p.a;extrapolation_bc=Line())
            for i in eachindex(p.a)
                ap = p.a[i] <= a_endo[1] ? first(p.a) : inverse(p.a[i])
                resources = (1+r)*p.a[i]+w*p.income[z]
                policy[i,z] = clamp(ap,first(p.a),min(last(p.a),resources-1e-12))
                cnew[i,z] = resources-policy[i,z]
            end
        end
        residual = maximum(abs.(cnew-c))
        c,cnew = cnew,c
        residual < tol && return (;policy,c,iterations=iteration,residual)
    end
    error("Endogenous grid iteration did not converge")
end

# Sparse T[to,from] contains the same Young probabilities as the illustrative loop.
# This avoids both repeated searches and an enormous (state,action,next-state) array.
function stationary_distribution_fast(policy,grid,P;initial=nothing,tol=1e-10,maxiter=100_000)
    na,nz = size(policy)
    rows,cols,weights = Int[],Int[],Float64[]
    sizehint!(rows,2na*nz^2); sizehint!(cols,2na*nz^2); sizehint!(weights,2na*nz^2)
    for z in 1:nz,i in 1:na
        j,w = lottery(grid,policy[i,z])
        for zp in 1:nz
            append!(rows,(j+(zp-1)*na,j+1+(zp-1)*na))
            append!(cols,(i+(z-1)*na,i+(z-1)*na))
            append!(weights,((1-w)*P[z,zp],w*P[z,zp]))
        end
    end
    transition = sparse(rows,cols,weights,na*nz,na*nz)
    dropzeros!(transition)
    mu = isnothing(initial) ? fill(1/(na*nz),na*nz) : vec(copy(initial))
    next = similar(mu)
    for iteration in 1:maxiter
        mul!(next,transition,mu)
        residual = 0.0
        for i in eachindex(mu)
            residual += abs(next[i]-mu[i])
        end
        mu,next = next,mu
        residual < tol && return (;mu=reshape(mu,na,nz),iterations=iteration,residual,transition)
    end
    error("Distribution iteration did not converge")
end

solve(p=parameters(na=800,nd=5000);kwargs...) = solve_equilibrium(p;
    household_solver=solve_household_egm,distribution_solver=stationary_distribution_fast,
    warm_start=true,kwargs...)

function make_figure(result,p;path=joinpath(@__DIR__,"aiyagari.png"))
    ENV["GKSwstype"] = "100"
    mass = vec(sum(result.distribution.mu;dims=2))
    upper = p.d[searchsortedfirst(cumsum(mass),0.9999)]
    edges = collect(range(first(p.d),last(p.d);length=101))
    bins = zeros(100)
    for (a,m) in zip(p.d,mass)
        bins[clamp(searchsortedlast(edges,a),1,100)] += m
    end
    default(fontfamily="sans-serif",dpi=180,guidefontsize=11,tickfontsize=10,
            titlefontsize=14,legendfontsize=10,gridalpha=0.15,
            left_margin=5Plots.mm,bottom_margin=5Plots.mm)
    labels = [@sprintf("Productivity %.1f",z) for z in p.income]
    policy = plot(p.a,result.household.policy;label=permutedims(labels),linewidth=2,
        xlabel="Current assets",ylabel="Next-period assets",title="Saving decisions",
        xlims=(0,upper),ylims=(0,upper))
    plot!(policy,p.a,p.a;label="45-degree line",linestyle=:dash,color=:gray)
    distribution = bar((edges[1:end-1]+edges[2:end])/2,100bins;
        bar_width=diff(edges)[1],linewidth=0,color="#345b7d",label=false,
        xlabel="Assets",ylabel="Population per bin (%)",
        title=@sprintf("Stationary assets | %.1f%% at zero assets",100mass[1]),xlims=(-0.1,upper+0.2))
    figure = plot(policy,distribution;layout=(1,2),size=(1300,520),
        plot_title=@sprintf("Aiyagari | interest rate %.2f%% | aggregate capital %.3f",100result.r,result.capital))
    savefig(figure,path)
    path
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    using .AiyagariEfficient
    p = parameters(na=800,nd=5000)
    @time result = solve(p)
    println("Capital: ",result.capital," | wage: ",result.w)
    println("Figure: ",make_figure(result,p))
end
