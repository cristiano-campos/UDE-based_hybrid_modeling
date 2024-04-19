using Distributions
using Random
using BSON
using OrdinaryDiffEq
using Suppressor
using LinearAlgebra

#the big model 
function big_model!(du, u, p, t)
    
    x1,    x2, x1x2,    x3,  x4, x3x4,     x5, x3x5,   x6,   x2x6,    x7, 
    U1,    U2, U1U2,    U3,  U4, U3U4,     U5,   U6, U5U6,     U7,    U8, 
  U7U8,    U9,  U10, U9U10, U11,  U12, U11U12,  U13,  U14, U13U14, U6U14, 
  U4U7, U9U12,U8U11, x6U13, x7U7 = u
  
  kf1, kr1, kcat1, 
  kf2, kr2, kcat2,
  kf3, kr3, kcat3,
  kf4, kr4, kcat4, 
  kf5, kr5, kcat5, 
  kf6, kr6, kcat6, 
  kf7, kr7, kcat7, 
  kf8, kr8, kcat8, 
  kf9, kr9, kcat9, 
  kf10, kr10, kcat10, 
  kf11, kr11, kcat11, 
  kf12, kr12, kcat12,
  kf13, kr13, kcat13, 
  kf14, kr14, kcat14,
  kf15, kr15, kcat15,
  kf16, kr16, kcat16,
  kf17, kr17, kcat17 = p
  
  #  x1
  du[1]  = -1 * (kf1 * x1 * x2)  + (kr1 * x1x2 + kcat1 * x1x2 + kcat5 * U1U2);
  #  x2
  du[2]  = -1 * (kf1 * x1 * x2 + kf4 * x2 * x6) + (kr1 * x1x2 + kr4 * x2x6 + kcat6 * U3U4);
  #  x1x2
  du[3]  = -1 * (kcat1 * x1x2 + kr1 * x1x2)  + (kf1 * x1 * x2);
  #  x3
  du[4]  = -1 * (kf2 * x3 * x4 + kf3 * x3 * x5) + (kcat1 * x1x2 + kcat2 * x3x4 + kcat3 * x3x5 + kr2 * x3x4 + kr3 * x3x5 + kcat7 * U11U12);
  #  x4
  du[5]  = -1 * (kf2 * x3 * x4) + (kr2 * x3x4 + kcat8 * U9U10);
  #  x3x4
  du[6]  = -1 * (kcat2 * x3x4 + kr2 * x3x4) + (kf2 * x3 * x4);
  #  x5
  du[7]  = -1 * (kf3 * x3 * x5) + (kcat2 * x3x4 + kr3 * x3x5 + kcat9 * U13U14);
  #  x3x5
  du[8]  = -1 * (kcat3 * x3x5 + kr3 * x3x5) + (kf3 * x3 * x5);
  #  x6
  du[9]  = -1 * (kf4 * x2 * x6 + kf16 * x6 * U13) + (kcat3 * x3x5 + kcat4 * x2x6 + kr4 * x2x6 + kcat10 * U5U6 + kr16 * x6U13);
  #  x2x6
  du[10] = -1 * (kcat4 * x2x6 + kr4 * x2x6) + (kf4 * x2 * x6);
  #  x7
  du[11] = -1 * (kf17 * x7 * U7) + (kcat11 * U7U8 + kcat4 * x2x6 + kr17 * x7U7);
  #  U1
  du[12] = -1 * (kf5 * U1 * U2) + (kr5 * U1U2 + kcat13 * U4U7);
  #  U2
  du[13] = -1 * (kf5 * U1 * U2) + (kcat5 * U1U2 + kr5 * U1U2);
  #  U1U2
  du[14] = -1 * (kcat5 * U1U2 + kr5 * U1U2) + (kf5 * U1 * U2);
  #  U3
  du[15] = -1 * (kf6 * U3 * U4) + (kr6 * U3U4 + kcat14 * U9U12);
  #  U4
  du[16] = -1 * (kf6 * U3 * U4 + kf13 * U4 * U7) + (kcat6 * U3U4 + kr6 * U3U4 + kr13 * U4U7 + kcat13 * U4U7);
  #  U3U4
  du[17] = -1 * (kcat6 * U3U4 + kr6 * U3U4) + (kf6 * U3 * U4);
  #  U5
  du[18] = -1 * (kf10 * U5 * U6) + (kcat10 * U5U6 + kr10 * U5U6);
  #  U6
  du[19] = -1 * (kf10 * U5 * U6) + (kr10 * U5U6 + kr12 * U6U14 + kcat15 * U8U11 + kcat17 * x7U7);
  #  U5U6
  du[20] = -1 * (kcat10 * U5U6 + kr10 * U5U6) + (kf10 * U5 * U6);
  #  U7
  du[21] = -1 * (kf11 * U7 * U8 + kf13 * U4 * U7 + kf17 * x7 * U7) + (kcat11 * U7U8 + kr11 * U7U8 + kr13 * U4U7 + kr17 * x7U7 + kcat17 * x7U7);
  #  U8
  du[22] = -1 * (kf11 * U7 * U8 + kf15 * U8 * U11) + (kr11 * U7U8 + kr15 * U8U11);
  #  U7U8
  du[23] = -1 * (kcat11 * U7U8 + kr11 * U7U8) + (kf11 * U7 * U8);
  #  U9
  du[24] = -1 * (kf8 * U10 * U9 + kf14 * U9 * U12) + (kcat8 * U9U10 + kr8 * U9U10 + kr14 * U9U12 + kcat14 * U9U12);
  #  U10
  du[25] = -1 * (kf8 * U10 * U9) + (kr8 * U9U10 + kcat12 * U6U14);
  #  U9U10
  du[26] = -1 * (kcat8 * U9U10 + kr8 * U9U10) + (kf8 * U10 * U9);
  #  U11
  du[27] = -1 * (kf7 * U11 * U12 + kf15 * U8 * U11) + (kr7 * U11U12 + kr15 * U8U11 + kcat15 * U8U11);
  #  U12
  du[28] = -1 * (kf7 * U11 * U12 + kf14 * U9 * U12) + (kcat7 * U11U12 + kr7 * U11U12 + kr14 * U9U12 + kcat16 * x6U13);
  #  U11U12
  du[29] = -1 * (kcat7 * U11U12 + kr7 * U11U12) + (kf7 * U11 * U12);
  #  U13
  du[30] = -1 * (kf9 * U13 * U14 + kf16 * x6 * U13) + (kcat9 * U13U14 + kr9 * U13U14 + kcat16 * x6U13 + kr16 * x6U13);
  #  U14
  du[31] = -1 * (kf9 * U13 * U14) + (kr9 * U13U14 + kr12 * U6U14 + kcat12 * U6U14);
  #  U13U14
  du[32] = -1 * (kcat9 * U13U14 + kr9 * U13U14) + (kf9 * U13 * U14);
  
  # U6U14
  du[33] = -1 * (kcat12 * U6U14 + kr12 * U6U14) + (kf12 * U6 * U14);
  # U4U7
  du[34] = -1 * (kcat13 * U4U7 + kr13 * U4U7) + (kf13 * U4 * U7);
  # U9U12
  du[35] = -1 * (kcat14 * U9U12 + kr14 * U9U12) + (kf14 * U9 * U12);
  # U8U11
  du[36] = -1 * (kcat15 * U8U11 + kr15 * U8U11) + (kf15 * U8 * U11);
  
  # x6U13
  du[37] = -1 * (kcat16 * x6U13 + kr16 * x6U13) + (kf16 * x6 * U13);
  # x7U7
  du[38] = -1 * (kcat17 * x7U7 + kr17 * x7U7) + (kf17 * x7 * U7);
end;

#the first-principle bigmodel parameters 
bigmodel_param = Float32.([
    0.015, # kf1,   
    0.100, # kr1, 
    0.003, # kcat1, 

    0.099,   # kf2,  
    0.115,   # kr2, 
    0.085,   # kcat2,

    0.089,   # kf3,  
    0.05,    # kr3,
    0.15,    # kcat3, 
        
    0.25,    # kf4,  
    0.4325,  # kr4, 
    0.0150,   # kcat4,
        
    0.25,    # kf5, 
    0.35,    # kr5, 
    0.25,    # kcat5,
        
    0.25,    # kf6, 
    0.01,    # kr6, 
    0.45,    # kcat6,
        
    0.15,  # kf7, 
    0.05,  # kr7, 
    0.115, # kcat7, 
        
    0.15,  # kf8, 
    0.25,  # kr8, 
    0.315, # kcat8,
        
    0.10,  # kf9, 
    0.10,  # kr9,
    0.205, # kcat9,
        
    0.100,  # kf10,
    0.250,  # kr10,
    0.185, # kcat10, 
        
    0.05,  # kf11, 
    0.470,  # kr11,
    0.205,  # kcat11
        
    0.05,  # kf12, 
    0.100, # kr12,
    0.150,  # kcat12
        
    0.05,  # kf13, 
    0.056, # kr13,
    0.187,  # kcat13
        
    0.15,  # kf14, 
    0.105, # kr14,
    0.195,  # kcat14
        
    0.015,  # kf15, 
    0.025, # kr15,
    0.105,  # kcat15
        
    0.015,  # kf16, 
    0.025, # kr16,
    0.105,  # kcat16
        
    0.0250,   # kf17, 
    0.012,  # kr17,
    0.235  # kcat17
]);

species = [
     "x1",     "x2",  "x1x2",   "x3",    "x4",  "x3x4",   "x5", "x3x5",     "x6", "x2x6", 
     "x7",     "U1",    "U2", "U1U2",    "U3",    "U4", "U3U4",   "U5",     "U6", "U5U6", 
     "U7",     "U8",  "U7U8",   "U9",   "U10", "U9U10",  "U11",  "U12", "U11U12",  "U13", 
    "U14", "U13U14", "U6U14", "U4U7", "U9U12", "U8U11", "x6U13", "x7U7"
];

tspan = (0.0f0, 100.0f0);

#the true indicates time-idependency 
#SciMLBase.FullSpecialize is a type indicating that the solver should fully specialize the problem 
prob = ODEProblem{true, SciMLBase.FullSpecialize}(big_model!, zeros(Float32, 38), tspan, bigmodel_param);

saveat = 1;
points_per_species = Int64(tspan[2]/saveat + 1);

n_train_size = 1000;
n_test_size = 1000;
n_val_size = 1000;

N = n_train_size + n_test_size + n_val_size; 

folder = "conditionsnoise_$(points_per_species)pts" 

mkpath(folder)

#this function sample the initial values of the model
function sample_u0(rng) 
    #d is a continuous distributions with a lower bound of 0 and upper bound of 13
    d = Uniform(0, 13);
    #It generates a random sample of size 38 from the uniform distribution d 
    sample = rand(rng, d, 38);
    return sample;
end

function simulate(_u0, model)
    tmp_prob = remake(model, u0 = _u0, p=bigmodel_param);
    return @suppress_err begin
        solve(
            tmp_prob,
            Vern7(),
            abstol=1e-12,
            reltol=1e-12,
            saveat = saveat 
        );
    end
end

selected_species = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

#5% percent of noise 
noise_magnitude = 5e-3

for condition in 1:N
    attempt = 1;

    println("Simulating Condition: $(condition)\n");
    println("\tAttempt: $(attempt)\r");

    #create a random number generator 
    rng = MersenneTwister(condition);
    _u0 = sample_u0(rng);
    sol = simulate(_u0, prob);
    
    while sol.retcode != :Success
        attempt += 1;
        println("\tAttempt: $(attempt)\r");
        _u0 = sample_u0(rng);
        sol = simulate(_u0, prob);
    end

    X = Array(sol)[selected_species, : ];

    # Add noise
    x̄ = mean(X, dims=2)
    noise = (noise_magnitude * x̄) .* randn(eltype(X), size(X))
    Xₙ = X .+ noise
    
    # Truncate negative values
    Xₙ[Xₙ .< 0] .= 0


    print("\tSavin to bson\n");

    if condition <= n_train_size
        filename = "train_condition_$(condition).bson";
    elseif condition <= n_train_size + n_val_size;

        filename = "test_condition_$(condition - n_train_size).bson";
    else 
        filename = "val_condition_$(condition - (n_train_size + n_val_size)).bson";
    end 
    #mkpath("($folde)/($filename)");
    bson("$(folder)/$(filename)", Dict(:u0 => _u0[selected_species],:X => Xₙ));
end