# Krusell-Smith: a readable approximate-aggregation loop with aggregate risk.
# Run: julia KrusellSmith_illustrative.jl
module KrusellSmithIllustrative

using LinearAlgebra, Statistics, Random, Printf
using Optim

export parameters, bracket, interpolate, prices, forecast, aggregate_path,
       solve_household, simulate, fit_law, forecast_errors, solve_equilibrium

function parameters(;na=70,nd=500,nK=5,amax=200.0,Kmin=25.0,Kmax=50.0,
                    periods=2400,burn=400,seed=1998)
    na>=3 && nd>=3 && nK>=2 || error("Grids are too small")
    periods>burn>=0 || error("Need observations after burn-in")
    0<Kmin<Kmax<amax || error("Invalid capital or asset bounds")
    a = amax.*collect(range(0,1;length=na)).^2
    d = amax.*collect(range(0,1;length=nd)).^2
    K = collect(range(Kmin,Kmax;length=nK))
    z,u = [0.99,1.01],[0.10,0.04] # bad/good productivity and unemployment
    Pz = [0.875 0.125;0.125 0.875]
    # Employment: 1 = unemployed, 2 = employed. Rows are today's states.
    # Conditional unemployment persistence for each (z,z') pair.
    stay_unemployed = [0.6 0.25;0.75 1/3]
    Pe = zeros(2,2,2,2) # Pe[e,e',z,z'] conditional on the aggregate transition
    for iz in 1:2,jz in 1:2
        p00 = stay_unemployed[iz,jz]
        p10 = (u[jz]-u[iz]*p00)/(1-u[iz])
        Pe[:,:,iz,jz] = [p00 1-p00;p10 1-p10]
    end
    # A teaching variant: common beta, unit hours, balanced-budget benefits.
    (;a,d,K,z,u,Pz,Pe,beta=0.99,alpha=0.36,delta=0.025,benefit=0.15,
      periods,burn,seed)
end

# Return adjacent nodes and the weight on the upper node. Reject extrapolation.
@inline function bracket(grid,x)
    first(grid)-1e-10<=x<=last(grid)+1e-10 || error("Point $x outside grid")
    j = clamp(searchsortedlast(grid,x),1,length(grid)-1)
    w = clamp((x-grid[j])/(grid[j+1]-grid[j]),0.0,1.0)
    j,w
end
@inline interpolate(values,j,w) = (1-w)*values[j]+w*values[j+1]

function prices(K,z,p)
    L = 1-p.u[z]
    R = 1-p.delta+p.alpha*p.z[z]*(K/L)^(p.alpha-1)
    wage = (1-p.alpha)*p.z[z]*(K/L)^p.alpha
    tax = p.benefit*p.u[z]/L
    income = (p.benefit*wage,(1-tax)*wage)
    (;R,wage,income,L,tax)
end
forecast(B,K,z) = exp(B[z,1]+B[z,2]*log(K))

# V(a,e,K,z): guess the capital law, then solve the household's Bellman problem.
# Scalar maximization keeps a' continuous; ten fixed-policy evaluation sweeps
# (Howard steps) save time without changing the Bellman problem.
function solve_household(B,p;initial=nothing,tol=2e-7,maxiter=3000,howard=10)
    resources = [prices(K,z,p).R*a+prices(K,z,p).income[e]
                 for a in p.a,e in 1:2,K in p.K,z in 1:2]
    V = isnothing(initial) ? log.(resources)/(1-p.beta) : copy(initial.V)
    Vnew,policy,EV = similar(V),similar(V),similar(V)
    for iteration in 1:maxiter
        improve = mod(iteration-1,howard+1)==0
        for z in 1:2,k in eachindex(p.K)
            kp,wk = bracket(p.K,forecast(B,p.K[k],z))
            for e in 1:2,i in eachindex(p.a)
                EV[i,e,k,z] = sum(p.Pz[z,zp]*p.Pe[e,ep,z,zp]*
                    ((1-wk)*V[i,ep,kp,zp]+wk*V[i,ep,kp+1,zp])
                    for ep in 1:2,zp in 1:2)
            end
            for e in 1:2,i in eachindex(p.a)
                cash = resources[i,e,k,z]
                ev = @view EV[:,e,k,z]
                value(ap) = log(cash-ap)+p.beta*interpolate(ev,bracket(p.a,ap)...)
                if improve
                    upper = min(last(p.a),cash-1e-10)
                    optimum = optimize(ap -> -value(ap),0.0,upper)
                    choices = (0.0,Optim.minimizer(optimum),upper)
                    policy[i,e,k,z] = choices[argmax(value.(choices))]
                end
                Vnew[i,e,k,z] = value(policy[i,e,k,z])
            end
        end
        residual = maximum(abs.(Vnew-V))
        V,Vnew = Vnew,V
        if improve && residual<tol
            return (;policy,c=resources-policy,V,iterations=iteration,residual)
        end
    end
    error("Household VFI did not converge")
end

function aggregate_path(p;seed=p.seed,periods=p.periods)
    rng = MersenneTwister(seed)
    z = ones(Int,periods+1)
    for t in 1:periods
        z[t+1] = rand(rng)<p.Pz[z[t],1] ? 1 : 2
    end
    z
end

# Simulate only aggregate shocks. Transport an entire population distribution
# with Young lotteries; do NOT draw a noisy panel of individual households.
# Crucially use Pe conditional on the REALIZED z -> z', not the joint matrix.
function simulate(policy,p,zpath;initial=nothing)
    nd,T = length(p.d),length(zpath)-1
    T>p.burn || error("Simulation must extend beyond burn-in")
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
        K[t] = sum(mu.*p.d)
        k,wk = bracket(p.K,K[t])
        fill!(next,0.0)
        t>p.burn && (average_mu .+= mu)
        pr = prices(K[t],z,p)
        for e in 1:2,i in 1:nd
            ia,wa = bracket(p.a,p.d[i])
            ap = (1-wk)*interpolate(@view(policy[:,e,k,z]),ia,wa)+
                  wk*interpolate(@view(policy[:,e,k+1,z]),ia,wa)
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

function fit_law(sim,p)
    B,r2 = zeros(2,2),zeros(2)
    for z in 1:2
        times = [t for t in p.burn+1:length(sim.K)-1 if sim.z[t]==z]
        length(times)>10 || error("Too few observations for aggregate state $z")
        x,y = log.(sim.K[times]),log.(sim.K[times.+1])
        X = hcat(ones(length(times)),x)
        B[z,:] = X\y
        r2[z] = 1-sum(abs2,y-X*B[z,:])/sum(abs2,y.-mean(y))
    end
    (;B,r2)
end

# One-step errors use actual K_t, NOT a recursively forecast capital series.
function forecast_errors(B,sim,p)
    times = p.burn+1:length(sim.K)-1
    error = [100*(forecast(B,sim.K[t],sim.z[t])/sim.K[t+1]-1) for t in times]
    (;rmse=sqrt(mean(abs2,error)),maximum=maximum(abs.(error)),errors=error)
end

function solve_equilibrium(p=parameters();household_solver=solve_household,
                           simulator=simulate,tol=2e-5,maxiter=100,damping=0.25,
                           initial_B=nothing,verbose=true)
    0<damping<=1 || error("Damping must be in (0,1]")
    B = isnothing(initial_B) ? [0.04log(mean(p.K)) 0.96;0.04log(mean(p.K)) 0.96] : copy(initial_B)
    zpath,household = aggregate_path(p),nothing
    for iteration in 1:maxiter
        household = household_solver(B,p;initial=household)
        sim = simulator(household.policy,p,zpath)
        fit = fit_law(sim,p)
        residual = maximum(abs.(fit.B-B))
        verbose && @printf("KS %2d | coefficient gap %.3e | R2 %.7f / %.7f\n",
                           iteration,residual,fit.r2...)
        if residual<tol
            # Return the law actually used by households, not an unsolved update.
            return (;B,household,simulation=sim,fit,iterations=iteration,residual,
                    errors=forecast_errors(B,sim,p))
        end
        B = (1-damping)*B+damping*fit.B
    end
    error("Capital forecasting law did not converge")
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    using .KrusellSmithIllustrative
    @time result = solve_equilibrium()
    println("Rows bad/good; columns intercept/slope:\n",result.B)
    println("One-step capital forecast RMSE (%): ",result.errors.rmse)
end
