# Bradley J. Setzler
# JuliaEconomics.com
# Tutorial 6: Kalman Filter for Panel Data and MLE in Julia
# Passed test on Julia 0.4, but is now much slower

using DataFrames
using Distributions
using Optim

function unpackParams(params,stateDim,obsDim)
    place = 0
    A = reshape(params[(place+1):(place+stateDim^2)],(stateDim,stateDim))
    place += stateDim^2
    V = diagm(exp(params[(place+1):(place+stateDim)]))
    place += stateDim
    Cparams = params[(place+1):(place+stateDim*(obsDim-1))]
    C = zeros(stateDim*obsDim,stateDim)
    for j in [1:stateDim]
            C[(1+obsDim*(j-1)):obsDim*j,j] = [1,Cparams[(1+(obsDim-1)*(j-1)):(obsDim-1)*j]]
    end
    place += (obsDim-1)*stateDim
    W = diagm(exp(params[(place+1):(place+obsDim*stateDim)]))
    return ["A"=>A,"V"=>V,"C"=>C,"W"=>W]
end


function KalmanDGP(params,stateDim,obsDim,N,T,init_exp,init_var)
    # initialize data
    data = zeros(N,stateDim*obsDim*T+1)
    # parameters
    unpacked = unpackParams(params,stateDim,obsDim)
    A = unpacked["A"]
    V = unpacked["V"]
    C = unpacked["C"]
    W = unpacked["W"]
    # draw from DGP
    for i=1:N
        # data of individual i
        iData = ones(stateDim*obsDim*T+1)
        current_state = rand(MvNormal(reshape(init_exp,(stateDim,)),init_var))
        iData[1:stateDim*obsDim] = rand(MvNormal(eye(obsDim*stateDim)))
        for t=2:T
            current_state = A*current_state + rand(MvNormal(V))
            iData[((t-1)*stateDim*obsDim+1):(t*stateDim*obsDim)] = C*current_state + rand(MvNormal(W))
        end
        # add individual to data
        data[i,:] = iData
    end
    return data
end


function incrementKF(params,post_exp,post_var,new_obs,stateDim,obsDim)
    # unpack parameters
    unpacked = unpackParams(params,stateDim,obsDim)
    A = unpacked["A"]
    V = unpacked["V"]
    C = unpacked["C"]
    W = unpacked["W"]
    # predict
    prior_exp = A*post_exp
    prior_var = A*post_var*A' + V
    obs_prior_exp = C*prior_exp
    obs_prior_var = C*prior_var*C' + W
    # update
    residual = new_obs - obs_prior_exp
    obs_prior_cov = prior_var*C'
    kalman_gain = obs_prior_cov*inv(obs_prior_var)
    post_exp = prior_exp + kalman_gain*residual
    post_var = prior_var - kalman_gain*obs_prior_cov'
    # step likelihood
    dist = MvNormal(reshape(obs_prior_exp,(length(obs_prior_exp),)),obs_prior_var)
    log_like = logpdf(dist,new_obs)
    return ["post_exp"=>post_exp,"post_var"=>post_var,"log_like"=>log_like]
end


function indivKF(params,df,obsDict,init_exp,init_var,stateDim,obsDim,T,i)
    iData = df[i,:]
    # initialization
    post_exp = init_exp
    post_var = init_var
    init_obs = array(iData[obsDict[1]])'
    dist = MvNormal(eye(length(init_obs)))
    log_like=logpdf(dist,init_obs)
    for t = 1:(T-1)
        # predict and update
        new_obs = array(iData[obsDict[t+1]])'
        new_post = incrementKF(params,post_exp,post_var,new_obs,stateDim,obsDim)
        # replace
        post_exp = new_post["post_exp"]
        post_var = new_post["post_var"]
        # contribute
        log_like += new_post["log_like"]
    end
    return log_like
end


function sampleKF(params,df,obsDict,init_exp,init_var,stateDim,obsDim,T)
    log_like = 0.0
    N = size(df,1)
    for i in 1:N
        log_like += indivKF(params,df,obsDict,init_exp,init_var,stateDim,obsDim,T,i)
    end
    neg_avg_log_like = -log_like/N
    println("current average negative log-likelihood: ",neg_avg_log_like)
    return neg_avg_log_like[1]
end









function estimateKF()
    srand(2)

    params0 = [1.,0.,0.,1.,0.,0.,.5,-.5,.5,-.5,0.,0.,0.,0.,0.,0.]

    stateDim = 2
    obsDim = 3
    T = 4
    N = 1000
    init_exp = zeros(stateDim)
    init_var = eye(stateDim)
    data=KalmanDGP(params0,stateDim,obsDim,N,T,init_exp,init_var)
    data = DataFrame(data)
    names!(data, [:one_1,:one_2,:one_3,:one_4,:one_5,:one_6,:two_1,:two_2,:two_3,:two_4,:two_5,:two_6,:three_1,:three_2,:three_3,:three_4,:three_5,:three_6,:four_1,:four_2,:four_3,:four_4,:four_5,:four_6,:outcome])
    obsDict = {[:one_1,:one_2,:one_3,:one_4,:one_5,:one_6],[:two_1,:two_2,:two_3,:two_4,:two_5,:two_6],[:three_1,:three_2,:three_3,:three_4,:three_5,:three_6],[:four_1,:four_2,:four_3,:four_4,:four_5,:four_6]}

    function wrapLoglike(params)
        print("current parameters: ",params)
        return sampleKF(params,data,obsDict,init_exp,init_var,stateDim,obsDim,T)
    end


    tic()
    MLE = optimize(wrapLoglike,params0,method=:cg,ftol=1e-8)
    toc()
    optimParams = unpackParams(MLE.minimum,stateDim,obsDim)

    return optimParams

end

tic()
optimParams0 = estimateKF()
toc()

© 2019 GitHub, Inc.
