using Distributions
using DifferentialEquations
using Random
using BSON
using OrdinaryDiffEq
using Suppressor
using LinearAlgebra

function big_model!(du, u, p, t )
    
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17 = u

    k1, k2, k3 , k4 , k5 , k6 , k7 , k8 , k9 , k10 , k11 , k12 , k13 , k14 , k15 , k16 , k17 , k18 , k19 , k20 , k21 , k22 , k23 , k24 , k25 , k26 , k27 , k28 , k29 , k30 , k31 , k33 , k32 = p

    du[1] = (-1)*x1*k3 + 1*x4*k6 + (-1)*x1*k8 + 1*x5*k9 + (-1)*x1*k10 + 1*x6*k11 + (-1)*x1*k12 + (-1)*x1*k13 + (-1)*x1*k14 + (-1)*x1*k17 + 1*x11*k18 + (-1)*x1*k19 + 1*x12*k20 + (-1)*x1*k21 + 1*x13*k22 + (-1)*x1*k23 + 1*x14*k24 + (-1)*x1*k25 + (-1)*x1*k26 + 1*x16*k27 + (-1)*x1*k28 + 1*x17*k29 + 1*k30*x7
    du[2] = 1*x1*k3 + (-1)*x2*k4 + (-1)*x2*k7
    du[3] = 1*x2*k4 + (-1)*x3*k5
    du[4] = 1*x3*k5 + (-1)*x4*k6 + 1*x2*k7
    du[5] = 1*x1*k8 + (-1)*x5*k9
    du[6] = 1*x1*k10 + (-1)*x6*k11
    du[7] = 1*x1*k12 + (-1)*k30*x7
    du[8] = 1*x1*k14 + (-1)*x8*k16
    du[9] = 1*x1*k13 + (-1)*x9*k15
    du[10] = 1*x9*k15 + 1*x8*k16 + 1*k31*x15
    du[11] = 1*x1*k17 + (-1)*x11*k18
    du[12] = 1*x1*k19 + (-1)*x12*k20
    du[13] = 1*x1*k21 + (-1)*x13*k22
    du[14] = 1*x1*k23 + (-1)*x14*k24
    du[15] = 1*x1*k25 + (-1)*k31*x15
    du[16] = 1*x1*k26 + (-1)*x16*k27
    du[17] = 1*x1*k28 + (-1)*x17*k29
end 

bigmodel_param = Float32.([ 1, 1, 661/50, 37/20, 3/100, 1461/100, 14/25, 227/100, 1/4, 24/25, 3/100, 1/50, 49/50, 26/25, 3/10, 3/100, 11/100, 3/50, 79/100, 41/100, 
21/50, 1/5, 1/25, 1/20, 9/100, 1/25, 1/10, 3/100, 1/50, 17/100, 9/50, 0, 100])

tspan = (0.0f0, 100.0f0)
saveat = 1

u0 = Float32.([100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

prob = ODEProblem{true, SciMLBase.FullSpecialize}(big_model!, zeros(Float32, 17), tspan, bigmodel_param)

saveat = 1
points_per_species = Int64(tspan[2]/saveat + 1)

n_train_size = 10
n_test_size = 10
n_val_size = 10

N = n_train_size + n_test_size + n_val_size

folder = "conditions_tiago_$(points_per_species)pts"

if isdir(folder)
    println("Folder $(folder) already exists");
else
    mkpath(folder)
end

function sample_u0(rng) 
    d = Uniform(0, 13)
    sample = rand(rng, d, 17)
    return sample
end

selected_species = [1, 2, 4, 11]

function simulate(_u0, model)
    tmp_prob = remake(model, u0 = _u0, p=bigmodel_param);
    return @suppress_err begin
        solve(
            tmp_prob,
            AutoTsit5(Rosenbrock23()),
            saveat = saveat 
        );
    end
end

for condition in 1:N
    attempt = 1

    print("Simulating Condition: $(condition)\n")
    print("\tAttempt: $(attempt)\r")
    
    rng    = MersenneTwister(condition)
    _u0    = sample_u0(rng)
    sol    = simulate(_u0, prob)

    while sol.retcode != :Success
        attempt += 1
        print("\tAttempt: $(attempt)\r")
        _u0  = sample_u0(rng)
        sol = simulate(_u0, prob)
    end

    X = Array(sol)[ selected_species , : ]
    
    print("\tSaving to bson\n")
    
    if condition <= n_train_size
        filename = "train_condition_$(condition).bson"
    elseif condition <= n_train_size + n_val_size  
        filename = "test_condition_$(condition - n_train_size).bson"
    else
        filename = "val_condition_$(condition - (n_train_size + n_val_size)).bson"
    end
    
    bson("$(folder)/$(filename)", Dict(:u0 => _u0[selected_species], :X => X))
end

