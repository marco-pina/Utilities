# Same Huggett economy; faster household and distribution algorithms.
# Run: julia Huggett_efficient.jl  (also writes huggett.png)
module HuggettEfficient

include("Huggett_illustrative.jl")
using .HuggettIllustrative
using Interpolations: linear_interpolation, Line
using LinearAlgebra, SparseArrays, Printf
using Plots

export parameters, solve_household_egm, stationary_distribution_fast, solve, make_figure

# EGM replaces maximization at every state with the Euler equation:
# q*c^(-gamma) = beta*E[c_next^(-gamma)].
function solve_household_egm(q,p; initial=nothing,tol=1e-9,maxiter=20_000)
    c = isnothing(initial) ? p.a .+ p.income' .- q*first(p.a) : copy(initial.c)
    policy,cnew = similar(c),similar(c)
    for iteration in 1:maxiter
        expected = c.^(-p.gamma)*p.P'
        c_endo = (p.beta/q .* expected).^(-1/p.gamma)
        for z in eachindex(p.income)
            a_endo = c_endo[:,z] .+ q.*p.a .- p.income[z]
            all(diff(a_endo).>0) || error("Endogenous grid is not increasing")
            inverse = linear_interpolation(a_endo,p.a;extrapolation_bc=Line())
            for i in eachindex(p.a)
                ap = p.a[i] <= a_endo[1] ? first(p.a) : inverse(p.a[i])
                upper = min(last(p.a),(p.a[i]+p.income[z]-1e-12)/q)
                policy[i,z] = clamp(ap,first(p.a),upper)
                cnew[i,z] = p.a[i]+p.income[z]-q*policy[i,z]
            end
        end
        residual = maximum(abs.(cnew-c))
        c,cnew = cnew,c
        residual < tol && return (;policy,c,iterations=iteration,residual)
    end
    error("Endogenous grid iteration did not converge")
end

# Precompute the Young transition once, then reuse sparse matrix multiplication.
# Column 'from' contains next-period probabilities, so next_mu = T*mu.
function stationary_distribution_fast(policy,grid,P; initial=nothing,tol=1e-10,maxiter=100_000)
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

function make_figure(result,p;path=joinpath(@__DIR__,"huggett.png"))
    ENV["GKSwstype"] = "100"
    mass = vec(sum(result.distribution.mu;dims=2))
    upper = p.d[searchsortedfirst(cumsum(mass),0.9999)]
    edges = collect(first(p.d):0.1:last(p.d))
    bins = zeros(length(edges)-1)
    for (a,m) in zip(p.d,mass)
        bins[clamp(searchsortedlast(edges,a),1,length(bins))] += m
    end
    default(fontfamily="sans-serif",dpi=180,guidefontsize=11,tickfontsize=10,
            titlefontsize=14,legendfontsize=10,gridalpha=0.15,
            left_margin=5Plots.mm,bottom_margin=5Plots.mm)
    ids = unique([1,cld(length(p.income),2),length(p.income)])
    labels = [@sprintf("Income %.2f",p.income[z]) for z in ids]
    policy = plot(p.a,result.household.policy[:,ids];label=permutedims(labels),
        linewidth=2,xlabel="Current assets",ylabel="Next-period assets",
        title="Saving decisions",xlims=(first(p.a),upper),ylims=(first(p.a),upper))
    plot!(policy,p.a,p.a;label="45-degree line",linestyle=:dash,color=:gray)
    distribution = bar((edges[1:end-1]+edges[2:end])/2,100bins;
        bar_width=diff(edges)[1],linewidth=0,color="#345b7d",label=false,
        xlabel="Assets",ylabel="Population per bin (%)",
        title=@sprintf("Stationary assets | %.1f%% at borrowing limit",100mass[1]),
        xlims=(first(p.d)-0.04,upper+0.1))
    figure = plot(policy,distribution;layout=(1,2),size=(1300,520),
        plot_title=@sprintf("Huggett | %d income states | bond price q = %.4f",length(p.income),result.q))
    savefig(figure,path)
    path
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    using .HuggettEfficient
    p = parameters(na=800,nd=5000)
    @time result = solve(p)
    println("Borrowing-limit share: ",sum(result.distribution.mu[1,:]))
    println("Figure: ",make_figure(result,p))
end
