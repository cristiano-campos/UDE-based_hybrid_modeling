using Random;
using StableRNGs;

using DifferentialEquations;
using SciMLSensitivity;

using ComponentArrays;
using Suppressor;
using Lux;
using ComponentArrays;
using NNlib: sigmoid, relu, leakyrelu;
using Flux.Losses: mae, mse;
using Statistics;
using JLSO;

function ude_dynamics!(du, u, θ, nn_st, t)

    missingInputs = neuralNetwork(u, θ, nn_st)[1];
    x1, x2, x3, x4, x5 = @view u[:];
    
    k50 =  1/1000
    k51 = 1/5
    k58 = 1/1000
    k59 = 1/5
    k53 = 1/2000
    k61 = 1/2000
    
    #x8
    du[1] = (-1)(k58*x2*x1 - k59*x5) + 1*k61*x4 + missingInputs[1]
    #x12
    du[2] = (-1)*(k50*x2*x4 - k51*x3)  + (-1)*(k58*x2*x1 - k59*x5) + missingInputs[2]
    #x27
    du[3] = 1*(k50*x2*x4 - k51*x3) + (-1)*k53*x3 + missingInputs[3]
    #x30
    du[4] = (-1)*(k50*x2*x4 - k51*x3) + (-1)*k61*x4 + missingInputs[4]
    #x5
    du[5] = 1*k53*x3 + 1*(k58*x2*x1 - k59*x5) + missingInputs[5]
end

function prob_func(prob, i, repeat)
    remake(prob, u0=conditions[i][:u0]);
end

function valprob_func(prob, i, repeat)
    remake(prob, u0=conditions[train_size + i][:u0]);
end

function testprob_func(prob, i, repeat)
    remake(prob, u0 = conditions[train_size + val_size + i][:u0]);
end

rng = Random.default_rng(); 

neuralNetwork = Chain(
    Dense(5 => 5, leakyrelu),
    Dense(5 => 4, leakyrelu),
    Dense(4 => 5)
);

#setting up NN for trainning 
nn_p, nn_st = Lux.setup(rng, neuralNetwork);
nn_p = ComponentVector{Float64}(nn_p);
nn_p = nn_p .* 0 + Float64(1e-4) * randn(StableRNG(2031), eltype(nn_p), size(nn_p)) 

u0      = zeros(Float32, 5);
tspan   = (0.0f0, 100.0f0);

odeFunction!(du, u, p, t) = ude_dynamics!(du, u, p, nn_st, t);
model       = ODEProblem{true, SciMLBase.FullSpecialize}(odeFunction!, u0, tspan, nn_p);

conditions  = Dict[];
n_fails     = 0;

conditions = Dict[];
n_fails = 0; 

colors = [:red, :blue, :brown, :magenta, :green, :orange, :navyblue, :black, :pink, :plum4, :gray, :gray, :gray, :gray, :gray, :gray, :gray, :gray, :gray, :gray,:gray, :gray, :gray, :gray, :gray, :gray, :gray, :gray, :gray, :gray, :gray, :gray];
species_name = [
    "x1",     "x2",  "x3", "x4", "x5"
];

savaAT = experimentConfig["savePointsAt"];
observedSpecies = (
    haskey(experimentConfig, "observedSpecies") && experimentConfig["observedSpecies"] isa Vector
) ? experimentConfig["observedSpecies"] : 1:5;

ensembleProblem = EnsembleProblem(model, prob_func = prob_func); 
valEnsembleProblem = EnsembleProblem(model, prob_func = valprob_func);
testEnsembleProblem = EnsembleProblem(model, prob_func = testprob_func);

set_conditions(values) = begin
    global conditions = values; 
end   


check_val_iter_index      = 100;

min_val_loss_i      = 1;
min_val_loss_v      = Inf;
last_index_callback = 0;

"""
    Callback of the optimizer
"""
function callback(p, l)

    global last_index_callback += 1;

    valLoss   = Float32.(validation_loss(p));
    testLoss  = Float32.(testset_loss(p));
    trainLoss = Float32.(l);

    open("$(results_folder)/valLoss.csv", "a+")   do io writedlm(io, valLoss, ',');   end;
    open("$(results_folder)/testLoss.csv", "a+")  do io writedlm(io, testLoss, ',');  end;
    open("$(results_folder)/trainLoss.csv", "a+") do io writedlm(io, trainLoss, ','); end;

    # check if the validation loss continues to increase after check_val_iter_index steps
    is_to_check_loss_val = last_index_callback > check_val_iter_index;

    # Saves the best parameter till validation loss increased.
    if (min_val_loss_v > valLoss)

        global min_val_loss_i = last_index_callback;
        global min_val_loss_v = valLoss;
        global check_val_iter_index = min_val_loss_i + 100;
        
        JLSO.save(
            "$(val_param_folder)/val_param.jlso", 
            :ude_parameters => p
        );
    end;

    is_early_stop = is_to_check_loss_val && valLoss > min_val_loss_v;
    
    if (is_early_stop) global check_val_iter_index = last_index_callback + 101; end

    print("[$(last_index_callback)] CheckValLoss[$(check_val_iter_index)] MinValLoss[$(min_val_loss_i)]: $(min_val_loss_v) -- Train: $(trainLoss), Validation: $(valLoss), Test: $(testLoss)\r");

    return is_early_stop;
end;

function predict(θ, initial_condition)
    tmp_prob = remake(model, u0=initial_condition, p=θ);

    tmp_sol = solve(
        tmp_prob,
        AutoTsit5(Rosenbrock23()),
        saveat = savaAT,
        save_idxs = observedSpecies
    );

    return tmp_sol;
end

function test_loss(p, condition)
    sol = predict(p, condition[:u0]);

    if SciMLBase.successful_retcode(sol.retcode)
        X_hat  = Array(sol);
        return mae( X_hat, condition[:X]); 
    else
        return Inf; 
    end
end

function loss(p)
    return evaluate_loss(p, ensembleProblem, train_size, 0, train_size);
end

function validation_loss(p)
    @suppress_err begin
        return evaluate_loss(p, valEnsembleProblem, val_size, train_size, val_size); 
    end
end

function testset_loss(p)
    @suppress_err begin
        return evaluate_loss(p, testEnsembleProblem, test_size, train_size + val_size, test_size);
    end
end

function evaluate_loss(p, ensembleProblem, trajectories, step, N)

    sim = solve(
        ensembleProblem, 
        AutoTsit5(Rosenbrock23()),
        EnsembleThreads(),
        saveat = savaAT,
        save_idxs = observedSpecies, 
        trajectories = trajectories,
        p = p
    );

    # L2 regularization for the weights
    error = convert(eltype(p), 1e-3) * sum(abs2, p);
    instability_penalty = 1e6 

    for i in 1:N
        condition = conditions[step+i];
        sol = sim[i];

        if SciMLBase.successful_retcode(sol.retcode)
            X_hat= Array(sol);
            error += mae( X_hat , condition[:X]);
        else
            error += instability_penalty;
        end
    end

    return error;  
end

function config_initial_weights()

    error = loss(nn_p); 

    println("----First test----");
 
    cont = 0;
    while error == Inf 
        println("----Try again----");
        cont += 1;
    
        global nn_p, nn_st = Lux.setup(rng, neuralNetwork);
        global nn_p = ComponentVector{Float64}(nn_p);
        #nn_p = nn_p .* 0 + Float64(1e-4) * randn(StableRNG(2031), eltype(nn_p), size(nn_p)) 

        #global params = ComponentArray(nn_p);
        
        error = loss(nn_p);

        if cont == 30
            open("$(results_folder)/initialWeigthInstability.csv", "a+")   do io writedlm(io, cont, ',');   end; 
            error("Numeric Instability");
        end
    end
    open("$(results_folder)/initialWeigthInstability.csv", "a+")   do io writedlm(io, cont, ',');   end;
    return nn_p;
end

println("----Experiment Config Loaded----");
