using Distributions
using DifferentialEquations
using Random
using BSON
using OrdinaryDiffEq
using Suppressor
using LinearAlgebra

function big_model!(du, u, p, t )
    
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = u

    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20 = p

    du[1] = ((-1)*k17*(k1*x1*x4 - k2*x6) + 1*k17*(k15*x11 - k16*x1*x5))/k17
    #Mp
    du[2] = (1*k17*k3*x6 + (-1)*k17*(k4*x2*x4 - k5*x7) + 1*(k10*x9 - k11*x2*x5) + (-1)*k17*(k12*x2*x5 - k13*x10))/k17
    du[3] = (1*k17*k6*x7 + (-1)*k17*(k7*x3*x5 - k8*x8))/k17
    #MAPKK
    du[4] = ((-1)*k17*(k1*x1*x4 - k2*x6) + 1*k17*k3*x6 + (-1)*k17*(k4*x2*x4 - k5*x7) + 1*k17*k6*x7)/k17
    du[5] = ((-1)*k17*(k7*x3*x5 - k8*x8) + 1*(k10*x9 - k11*x2*x5) + (-1)*k17*(k12*x2*x5 - k13*x10) + 1*k17*(k15*x11 - k16*x1*x5))/k17
    #M_MAPKK
    du[6] = (1*k17*(k1*x1*x4 - k2*x6) + (-1)*k17*k3*x6)/k17
    #Mp_MAPKK
    du[7] = (1*k17*(k4*x2*x4 - k5*x7) + (-1)*k17*k6*x7)/k17
    du[8] = (1*k17*(k7*x3*x5 - k8*x8) + (-1)*k17*k9*x8)/k17
    du[9] = (1*k17*k9*x8 + (-1)*(k10*x9 - k11*x2*x5))/k17
    du[10] = (1*k17*(k12*x2*x5 - k13*x10) + (-1)*k17*k14*x10)/k17
    du[11] = (1*k17*k14*x10 + (-1)*k17*(k15*x11 - k16*x1*x5))/k17

end

bigmodel_param = Float32.([1/50, 1, 1/100, 4/125, 1, 15, 9/200, 1, 23/250, 1, 1/100, 1/100, 1, 1/2, 43/500, 11/10000, 1, 500, 50, 100]) 

species = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11"]

tspan = (0.0f0, 100.0f0);

prob = ODEProblem{true, SciMLBase.FullSpecialize}(big_model!, zeros(Float32, 11), tspan, bigmodel_param);

saveat = 1;
points_per_species = Int64(tspan[2]/saveat + 1);

n_train_size = 1000;
n_test_size = 1000;
n_val_size = 1000;

N = n_train_size + n_test_size + n_val_size; 

folder = "conditions_markevich_$(points_per_species)pts"

if isdir(folder)
    println("Folder $(folder) already exists");
else
    mkpath(folder)
end

function sample_u0(rng) 
    #d is a continuous distributions with a lower bound of 0 and upper bound of 500
    d = Uniform(0, 500);
    #It generates a random sample of size 11 from the uniform distribution d 
    sample = rand(rng, d, 11);
    return sample;
end

selected_species = [2, 4, 6, 7]

function simulate(_u0, model)
    tmp_prob = remake(model, u0 = _u0, p=bigmodel_param);
    return @suppress_err begin
        solve(
            tmp_prob,
            Vern7(),
            saveat = saveat 
        );
    end
end

for condition in 1:N
    attempt = 1

    print("Simulating Condition: $(condition)\n")
    print("\tAttempt: $(attempt)\r")
    
    rng    = MersenneTwister(condition);
    _u0    = sample_u0(rng);
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
        filename = "train_condition_$(condition).bson";
    elseif condition <= n_train_size + n_val_size  
        filename = "test_condition_$(condition - n_train_size).bson";
    else
        filename = "val_condition_$(condition - (n_train_size + n_val_size)).bson";
    end;
    
    bson("$(folder)/$(filename)", Dict(:u0 => _u0[selected_species], :X => X))
end;

