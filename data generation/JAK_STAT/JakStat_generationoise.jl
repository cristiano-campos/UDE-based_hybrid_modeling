using Distributions
using DifferentialEquations
using Random
using BSON
using OrdinaryDiffEq
using Suppressor
using LinearAlgebra

function big_model!(du, u, p, t)

    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34 = u

    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31, k32, k33, k34, 
    k35, k36, k37, k38, k39, k40, k41, k42, k43, k44, k45, k46, k47, k48, k49, k50, k51, k52, k53, k54, k55, k56, k57, k58, k59, k60, k61, k62, k63, k64, k65, k66, k67, 
    k68, k69, k70, k71, k72, k73, k74, k75 = p

    du[1] = ((-1)*k1*(k3*x1*x2 - k4*x3) + (-1)*k1*(k71*k75*x1 - k72*x34))/k1
    du[2] = ((-1)*k1*(k3*x1*x2 - k4*x3) + (-1)*k1*(k73*x34*x2 - k74*x4))/k1
    du[3] = (1*k1*(k3*x1*x2 - k4*x3) + (-1)*k1*(k5*k75*x3 - k6*x4))/k1
    du[4] = (1*k1*(k5*k75*x3 - k6*x4) + (-2)*k1*(k7*x4*x4 - k8*x5) + 1*k1*(k73*x34*x2 - k74*x4))/k1
    du[5] = (1*k1*(k7*x4*x4 - k8*x5) + (-1)*k1*k9*x5 + 1*k1*k19*x13 + 1*k1*k52*x27 + 1*k1*k60*x32 + 1*k1*k63*x33)/k1
    du[6] = (1*k1*k9*x5 + (-1)*k1*(k10*x7*x6 - k11*x8) + 1*k1*k12*x8 + (-1)*k1*(k13*x6*x9 - k14*x10) + (-1)*k1*(k17*x6*x12 - k18*x13) + (-1)*k1*(k46*x25*x6 - k47*x26) + 1*k1*k64*x26)/k1
    du[7] = ((-1)*k1*(k10*x7*x6 - k11*x8) + 1*k1*k22*x15 + (-1)*k1*(k26*x7*x9 - k27*x16) + 1*k2*k39*x21 + (-1)*k1*(k48*x7*x26 - k49*x30) + 1*k1*k52*x27 + (-1)*k1*(k56*x7*x33 - k57*x27) + 1*k1*k60*x32)/k1
    du[8] = (1*k1*(k10*x7*x6 - k11*x8) + (-1)*k1*k12*x8 + (-1)*k1*(k58*x12*x8 - k59*x32) + 1*k1*k61*x30 + (-1)*k1*(k65*x25*x8 - k66*x30))/k1
    du[9] = (1*k1*k12*x8 + (-1)*k1*(k13*x6*x9 - k14*x10) + (-2)*k1*(k15*x9*x9 - k16*x11) + (-1)*k1*(k20*x14*x9 - k21*x15) + (-1)*k1*(k26*x7*x9 - k27*x16))/k1
    du[10] = 1*k1*(k13*x6*x9 - k14*x10)/k1
    du[11] = (1*k1*(k15*x9*x9 - k16*x11) + (-1)*k1*(k23*x14*x11 - k24*x28) + (-1)*k1*k28*x11)/k1
    du[12] = ((-1)*k1*(k17*x6*x12 - k18*x13) + 1*k1*k19*x13 + (-1)*k1*(k50*x12*x30 - k51*x27) + 1*k1*k52*x27 + (-1)*k1*(k54*x12*x26 - k55*x33) + (-1)*k1*(k58*x12*x8 - k59*x32) + 1*k1*k60*x32 + 1*k1*k63*x33)/k1
    du[13] = (1*k1*(k17*x6*x12 - k18*x13) + (-1)*k1*k19*x13 + 1*k1*k62*x33 + (-1)*k1*(k67*x25*x13 - k68*x33))/k1
    du[14] = ((-1)*k1*(k20*x14*x9 - k21*x15) + 1*k1*k22*x15 + (-1)*k1*(k23*x14*x11 - k24*x28) + 1*k1*k25*x28)/k1
    du[15] = (1*k1*(k20*x14*x9 - k21*x15) + (-1)*k1*k22*x15)/k1
    du[16] = (1*k1*k25*x28 + 1*k1*(k26*x7*x9 - k27*x16))/k1
    du[17] = (1*k1*k28*x11 + 1*k2*(k29*x18*x18 - k30*x17) + (-1)*k2*(k34*x19*x17 - k35*x29))/k2
    du[18] = ((-2)*k2*(k29*x18*x18 - k30*x17) + (-1)*k2*(k31*x19*x18 - k32*x20) + (-1)*k2*(k37*x21*x18 - k38*x22))/k2
    du[19] = ((-1)*k2*(k31*x19*x18 - k32*x20) + 1*k2*k33*x20 + (-1)*k2*(k34*x19*x17 - k35*x29) + 1*k2*k36*x29)/k2
    du[20] = (1*k2*(k31*x19*x18 - k32*x20) + (-1)*k2*k33*x20)/k2
    du[21] = (1*k2*k33*x20 + (-1)*k2*(k37*x21*x18 - k38*x22) + (-1)*k2*k39*x21)/k2
    du[22] = (1*k2*k36*x29 + 1*k2*(k37*x21*x18 - k38*x22))/k2
    du[23] = (1*k2*k40*x17/(k41 + x17) + (-1)*k2*k42*x23)/k2
    du[24] = (1*k2*k42*x23 + (-1)*k1*k44*x24)/k1
    du[25] = (1*k1*k43*x24 + (-1)*k1*k45*x25 + (-1)*k1*(k46*x25*x6 - k47*x26) + 1*k1*k52*x27 + 1*k1*k63*x33 + (-1)*k1*(k65*x25*x8 - k66*x30) + (-1)*k1*(k67*x25*x13 - k68*x33) + (-1)*k1*(k69*x25*x32 - k70*x27))/k1
    du[26] = (1*k1*(k46*x25*x6 - k47*x26) + (-1)*k1*(k48*x7*x26 - k49*x30) + (-1)*k1*(k54*x12*x26 - k55*x33) + (-1)*k1*k64*x26)/k1
    du[26] = (1*k1*(k46*x25*x6 - k47*x26) + (-1)*k1*(k48*x7*x26 - k49*x30) + (-1)*k1*(k54*x12*x26 - k55*x33) + (-1)*k1*k64*x26)/k1
    du[27] = (1*k1*(k50*x12*x30 - k51*x27) + (-1)*k1*k52*x27 + (-1)*k1*k53*x27 + 1*k1*(k56*x7*x33 - k57*x27) + 1*k1*(k69*x25*x32 - k70*x27))/k1
    du[28] = (1*k1*(k23*x14*x11 - k24*x28) + (-1)*k1*k25*x28)/k1
    du[29] = (1*k2*(k34*x19*x17 - k35*x29) + (-1)*k2*k36*x29)/k2
    du[30] = (1*k1*(k48*x7*x26 - k49*x30) + (-1)*k1*(k50*x12*x30 - k51*x27) + (-1)*k1*k61*x30 + 1*k1*(k65*x25*x8 - k66*x30))/k1
    du[31] = 0  
    du[32] = (1*k1*k53*x27 + 1*k1*(k58*x12*x8 - k59*x32) + (-1)*k1*k60*x32 + (-1)*k1*(k69*x25*x32 - k70*x27))/k1
    du[33] = (1*k1*(k54*x12*x26 - k55*x33) + (-1)*k1*(k56*x7*x33 - k57*x27) + (-1)*k1*k62*x33 + (-1)*k1*k63*x33 + 1*k1*(k67*x25*x13 - k68*x33))/k1
    du[34] = (1*k1*(k71*k75*x1 - k72*x34) + (-1)*k1*(k73*x34*x2 - k74*x4))/k1
end 

#the first-principle bigmodel parameters 
bigmodel_param = Float32.([1, 1, 1/10, 1/20, 1/50, 1/50, 1/25, 1/5, 1/200, 1/125, 4/5, 2/5, 1/200, 1/2, 1/50, 1/10, 1/1000, 1/5, 3/1000, 1/1000, 1/5, 3/1000, 
        1/1000, 1/5, 3/1000, 1/5000000, 1/5, 1/200, 1/50, 1/10, 1/1000, 1/5, 1/200, 1/1000, 1/5, 1/200, 1/5000000, 1/5, 1/20, 1/100, 400, 1/1000, 1/100, 1/2000, 1/2000, 
        1/50, 1/10, 1/125, 4/5, 1/1000, 1/5, 3/1000, 1/2000, 1/1000, 1/5, 1/125, 4/5, 1/1000, 1/5, 3/1000, 1/2000, 1/2000, 3/1000, 1/2000, 1/50, 1/10, 1/50, 1/10, 1/50, 
        1/10, 1/50, 1/50, 1/10, 1/20, 10, 10, 10, 2000, 0, 100, 50, 60]);

species = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", 
"x24", "x25", "x26", "x27", "x28", "x29", "x30", "x31", "x32", "x33", "x34"]

tspan = (0.0f0, 100.0f0);

prob = ODEProblem{true, SciMLBase.FullSpecialize}(big_model!, zeros(Float32, 34), tspan, bigmodel_param);

saveat = 1;
points_per_species = Int64(tspan[2]/saveat + 1);

n_train_size = 1000;
n_test_size = 1000;
n_val_size = 1000;

N = n_train_size + n_test_size + n_val_size; 

folder = "conditionsnoise_yamada_$(points_per_species)pts"

if isdir(folder)
    println("Folder $(folder) already exists");
else
    mkpath(folder)
end

function sample_u0(rng) 
    #d is a continuous distributions with a lower bound of 0 and upper bound of 13
    d = Uniform(0, 1000);
    #It generates a random sample of size 38 from the uniform distribution d 
    sample = rand(rng, d, 34);
    return sample;
end


selected_species = selected_species = [8, 12, 27, 30, 32]

function simulate(_u0, model)
    tmp_prob = remake(model, u0 = _u0, p=bigmodel_param);
    return @suppress_err begin
        solve(
            tmp_prob,
            Rosenbrock23(),
            saveat = saveat 
        );
    end
end

#5% percent of noise 
noise_magnitude = 5e-3

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

    X = Array(sol)[selected_species, :]
    
    # Add noise
    x̄ = mean(X, dims=2)
    noise = (noise_magnitude * x̄) .* randn(eltype(X), size(X))
    Xₙ = X .+ noise

    # Truncate negative values
    Xₙ[Xₙ .< 0] .= 0

    print("\tSaving to bson\n")
    
    if condition <= n_train_size
        filename = "train_condition_$(condition).bson";
    elseif condition <= n_train_size + n_val_size  
        filename = "test_condition_$(condition - n_train_size).bson";
    else
        filename = "val_condition_$(condition - (n_train_size + n_val_size)).bson";
    end;
    
    bson("$(folder)/$(filename)", Dict(:u0 => _u0[selected_species], :X => Xₙ))
end;