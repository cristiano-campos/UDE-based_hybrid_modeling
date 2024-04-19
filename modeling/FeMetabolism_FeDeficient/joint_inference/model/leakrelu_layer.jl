using Random, StableRNGs;

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

function ude_dynamics!(du, u, θ, nn_st, t, know_mech_p)

    θ_mech  = θ.mech_p;

    missingInputs = neuralNetwork(u, θ.nn_p, nn_st)[1];
    x1, x2, x3, x4 = @view u[:];

    #k3 =  661/50
    #k6 = 1461/100
    #k17= 11/100
    #k18 = 3/50

    k3  = :k3 in keys(θ_mech) ? θ_mech.k3 : know_mech_p.k3;
    k6  = :k6   in keys(θ_mech) ? θ_mech.k6   : know_mech_p.k6;
    k17  = :k17   in keys(θ_mech) ? θ_mech.k17   : know_mech_p.k17;
    k18  = :k18   in keys(θ_mech) ? θ_mech.k18   : know_mech_p.k18;
    
    du[1] = (-1)*x1*k3 + (-1)*x1*k17 + 1*x4*k18 + missingInputs[1]       
    du[2] = 1*x1*k3  + missingInputs[2]
    du[3] = (-1)*x3*k6 + missingInputs[3]
    du[4] = 1*x1*k17 + missingInputs[4]
end  


all_params = ["k3", "k6", "k17", "k18"]

default_value = [ 661/50, 1461/100, 11/100, 3/50];

infer_mask = [p in user_params for p in all_params];

mech_p_to_infer_str     = "";
mech_p_not_to_infer_str = "";

# Parameters to be infered
unknow_mech_p = nothing;
# Parameter that should not be infered
know_mech_p = nothing;

rng = Random.default_rng();
#Initialize the parameters choose by the user with random values
function initialize_mech_p()
    global mech_p_to_infer_str = "";

    for (i, param) in enumerate(all_params)
        if infer_mask[i]
            global mech_p_to_infer_str *= "$(param)=$(rand()),";
        elseif know_mech_p === nothing
            global mech_p_not_to_infer_str *= "$(param)=$(default_value[i]),";
        end
    end

    # Parameters to be infered
    global unknow_mech_p = ComponentArray(
        eval(Meta.parse("($(mech_p_to_infer_str))"))
    );

    if (know_mech_p === nothing)
        # Parameter that should not be infered
        global know_mech_p = ComponentArray(
            eval(Meta.parse("($(mech_p_not_to_infer_str))"))
        );
    end
end

println("Initializing mechinistic parameters");
initialize_mech_p();

println("Know mech parameters: $(know_mech_p)");
println("Unknow mech parameters: $(unknow_mech_p)");

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
    Dense(4 => 5, leakyrelu),
    Dense(5 => 5, leakyrelu),
    Dense(5 => 4, leakyrelu),
    Dense(4 => 4)
);

nn_p, nn_st = Lux.setup(rng, neuralNetwork);
nn_p = ComponentVector{Float64}(nn_p);
nn_p = nn_p .* 0 + Float64(1e-4) * randn(eltype(nn_p), size(nn_p))

params = ComponentArray(
    nn_p=nn_p, 
    mech_p=unknow_mech_p
);

u0 = zeros(Float32, 4);
tspan = (0.0f0, 100.0f0);

odeFunction!(du, u, p, t) = ude_dynamics!(du, u, p, nn_st, t, know_mech_p);

model = ODEProblem{true, SciMLBase.FullSpecialize}(odeFunction!, u0, tspan, params);

conditions = Dict[];
n_fails = 0; 

colors = [:red, :blue, :brown, :magenta, :green, :orange, :navyblue, :black, :pink, :plum4, :gray, :gray, :gray, :gray, :gray, :gray, :gray, :gray, :gray, :gray,:gray, :gray, :gray, :gray, :gray, :gray, :gray, :gray, :gray, :gray, :gray, :gray];
species_name = [
    "x1", "x2",  "x3", "x4"
];

savaAT = experimentConfig["savePointsAt"];
observedSpecies = (
    haskey(experimentConfig, "observedSpecies") && experimentConfig["observedSpecies"] isa Vector
) ? experimentConfig["observedSpecies"] : 1:4;

ensembleProblem = EnsembleProblem(model, prob_func = prob_func); 
valEnsembleProblem = EnsembleProblem(model, prob_func = valprob_func);
testEnsembleProblem = EnsembleProblem(model, prob_func = testprob_func);

set_conditions(values) = begin
    global conditions = values; 
end   
#print(conditions);
#this is a parameter specifying how often to check the validation loss, in terms of the number of iterations
check_val_iter      = 100;
#this  is keeping track of the iteration number where the minimum validation loss was observed
min_val_loss_i      = 1;
#This is initializing the minimum validation loss with a value of the positive infinity
min_val_loss_v      = Inf;
#This represents the iteration number for the last callback
last_index_callback = 0;
#this might be a parameter representing an extra number for the last callback
otp_extra_step      = 0;

function callback(p, l)

    global last_index_callback += 1; 

    valLoss = Float32.(validation_loss(p));
    testLoss = Float32.(testset_loss(p));
    trainLoss = Float32.(l); 
 
    open("$(results_folder)/valLoss.csv", "a+")   do io writedlm(io, valLoss, ',');   end;
    open("$(results_folder)/testLoss.csv", "a+")  do io writedlm(io, testLoss, ',');  end;
    open("$(results_folder)/trainLoss.csv", "a+") do io writedlm(io, trainLoss, ','); end;
    open("$(results_folder)/mechParametersPerIteration.csv", "a+") do io writedlm(io, p.mech_p[:]', ','); end;

    is_to_check_loss_val = (last_index_callback - min_val_loss_i) > check_val_iter + otp_extra_step;

    if (min_val_loss_v > valLoss)
        global min_val_loss_i = last_index_callback; 
        global min_val_loss_v = valLoss;
        global otp_extra_step = 0;

        try
            JLSO.save(
                "$(val_param_folder)/val_param.jlso",
                :ude_parameters => p
            );
        catch error
            open("save_error_log.txt", "a") do file
                write(file, "Error saving val_param.jlso : $error\\n");
            end
        end

    end

    print("[$(last_index_callback)] MV[$(min_val_loss_i)]: $(min_val_loss_v) -- Train: $(trainLoss), Validation: $(valLoss), Test: $(testLoss)\r");

    return is_to_check_loss_val && min_val_loss_v > trainLoss;
end

function predict(θ, initial_condition)
    tmp_prob = remake(model, u0=initial_condition, p=θ);

    tmp_sol = solve(
        tmp_prob,
        AutoTsit5(Rosenbrock23()),
        saveat = savaAT,
        save_idxs = observedSpecies,
        sensealg = SciMLSensitivity.QuadratureAdjoint(
            autojacvec=SciMLSensitivity.ReverseDiffVJP()
        )
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
        p = p;
    );

    # L2 regularization for the weights
    error = convert(eltype(p.nn_p), 1e-3) * sum(abs2, p.nn_p);
    error += sum(relu.(p.mech_p[:] * -100000)); # Penalize negative parameters for mechanistic parameters

    for i in 1:N
        condition = conditions[step+i];
        sol = sim[i];

        if SciMLBase.successful_retcode(sol.retcode)
            X_hat= Array(sol);
            error += mae( X_hat , condition[:X]);
        else
            return Inf;
        end
    end

    return error;  

end

function config_initial_weights()

    error = loss(params); 

    println("----First test----");
 
    cont = 0;
    while error == Inf 
        println("----Try again----");
        cont += 1;
        initialize_mech_p();

        println(unknow_mech_p);

        global nn_p, nn_st = Lux.setup(rng, neuralNetwork);
        global nn_p = ComponentVector{Float64}(nn_p);
        global nn_p = nn_p .* 0 + Float64(1e-4) * randn(StableRNG(2031), eltype(nn_p), size(nn_p))
        global params = ComponentArray(nn_p=nn_p, mech_p=unknow_mech_p);
        
        error = loss(params);

        if cont == 30
            open("$(results_folder)/initialWeigthInstability.csv", "a+")   do io writedlm(io, cont, ',');   end; 
            error("Numeric Instability");
        end
    end
    open("$(results_folder)/initialWeigthInstability.csv", "a+")   do io writedlm(io, cont, ',');   end;
    return params;
end

println("----Experiment Config Loaded----");