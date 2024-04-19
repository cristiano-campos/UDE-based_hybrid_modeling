println("Number of Threads: $(Threads.nthreads())")

using Random

using Suppressor
using Optimization, OptimizationOptimisers, OptimizationOptimJL

using BSON
using JSON
using TimerOutputs
using DelimitedFiles
using SciMLBase
using ComponentArrays
using Dates
using LineSearches
using JLSO
using Plots

include("./CommandLineParser.jl")
include("./PrintLoggingUtil.jl")

using .CommandLineParser

currentTime = Dates.format(now(), "ddmmyy_HHMMSS")

#parseCommandLine2 returns a JSON with command-line parameters
parsed_args = parseCommandLine2(); 

#folder in which the model and the experiment config is stored
folder_model = parsed_args["folder-model"];
#The base_folder consist of the directory path that contains the model and experiment config 
base_folder = "$(pwd())/$(folder_model)";

increase_train_size_factor = parsed_args["increase-train-size-by"];
#userParams = parsed_args["user-params"];
# println(userParams);
#transform a json file into julia data struct dict 
experimentConfig = JSON.parsefile("$(base_folder)/$(parsed_args["config"])");
#println(experimentConfig)

#folder where the initial conditions is contained 
condition_folder = "$(base_folder)/$(experimentConfig["conditionFolder"])";
println(condition_folder);

#user_params = eval(Meta.parse(userParams));
#results_folder = 
#str_user_parmas = join(user_params, '-');
results_folder="$(base_folder)/results/results_$(split(parsed_args["config"], ".")[1])-$(increase_train_size_factor)-$(Dates.format(now(), "ddmmyyHHMM"))";
println(results_folder);
plots_folder = "$(results_folder)/plots";
val_plots_folder = "$(results_folder)/val_plots";
val_param_folder = "$(results_folder)/val_param";
loss_file_name="$(base_folder)/$(experimentConfig["modelFile"])";
dataframe_folder = "$(results_folder)/dataframes";

train_size              = increase_train_size_factor * experimentConfig["trainSize"];
test_size               = experimentConfig["testSize"];
val_size                = experimentConfig["valSize"];
adam_maxiters           = experimentConfig["adamIterations"];
bfgs_maxiters           = experimentConfig["bfgsIterations"];
saveAt                  = experimentConfig["savePointsAt"];
plotScatter             = experimentConfig["plotScatter"];
n_conditions            = train_size + test_size + val_size;
increase_train_size_factor = nothing;

mkpath(results_folder);
mkpath(plots_folder);
mkpath(val_plots_folder);
mkpath(val_param_folder);
mkpath(dataframe_folder);
include(loss_file_name);

println("----ADAM [$(adam_maxiters)]----")
println("----BFGS [$(bfgs_maxiters)]----")

println("----Train      Size: $(train_size)----")
println("----Test       Size: $(test_size)----")
println("----Validation Size: $(val_size)----")

println("----Load conditions----");
conditions = Dict[];

for i in 1:train_size
    push!(conditions, BSON.load("$(condition_folder)/train_condition_$(i).bson"));
end

for i in 1:val_size
    push!(conditions, BSON.load("$(condition_folder)/val_condition_$(i).bson"));
end

for i in 1:test_size
    push!(conditions, BSON.load("$(condition_folder)/test_condition_$(i).bson"));
end

println("----Conditions loaded----");
println("----config initial weights----");

neuralNetworkInitialParameters = @suppress_err begin 
    p = config_initial_weights();
end

println("---initial weights configured---");
initial_loss = loss(neuralNetworkInitialParameters); 
println("----initial loss: $(initial_loss)----");

#print_sumary(train_size, val_size, test_size, test_loss, conditions, neuralNetworkInitialParameters);

optimizationFunction = Optimization.OptimizationFunction(
    (x, p) -> loss(x),
    Optimization.AutoForwardDiff()
);

optimizationProblem = Optimization.OptimizationProblem(
    optimizationFunction,
    ComponentVector(neuralNetworkInitialParameters)
);

learningRates = [0.001]; #0.05,0.01,
iterations    = [adam_maxiters]; #200,200,

neuralNetworkInitialParameters = nothing;
const timerOutput = TimerOutput();
res1 = nothing;

for i in 1:size(learningRates)[1]
    global res1 = @timeit timerOutput "ADAM_TRAINING" begin
        @suppress_err begin 
            Optimization.solve(
                optimizationProblem,
                ADAM(learningRates[i]),
                maxiters = iterations[i],
                callback = callback,
                progress = false
            );
        end;
    end;

    validationMinimizer = JLSO.load("$(val_param_folder)/val_param.jlso")[:ude_parameters];
    #print_sumary(train_size, val_size, test_size, test_loss, conditions, validationMinimizer);

    global optimizationProblem = Optimization.OptimizationProblem(
        optimizationFunction, 
        validationMinimizer
    );

    global otp_extra_step = 100;
end;

validationMinimizer = JLSO.load("$(val_param_folder)/val_param.jlso")[:ude_parameters]; 

global optimizationProblem = Optimization.OptimizationProblem(
    optimizationFunction, 
    validationMinimizer
);

JLSO.save("$(results_folder)/adam_parameters.jlso", :ude_parameters => res1.minimizer);

otp_extra_step = 50;

res2 = @timeit timerOutput "BFGS_TRAINING" begin
    @suppress_err begin 
        Optimization.solve(
            optimizationProblem,
            BFGS(initial_stepnorm=0.01f0, linesearch=LineSearches.BackTracking()),
            allow_f_increases = false,
            maxiters = bfgs_maxiters,
            callback = callback,
            progress = false
        );
    end;
end;

JLSO.save("$(results_folder)/bfgs_parameters.jlso", :ude_parameters => res2.minimizer);

validationMinimizer = JLSO.load("$(val_param_folder)/val_param.jlso")[:ude_parameters];

#print_sumary(train_size, val_size, test_size, test_loss, conditions, validationMinimizer);

show(timerOutput)
println();

open("$(results_folder)/times.json", "a") do io
    JSON.print(io, TimerOutputs.todict(timerOutput), 4);
end;

include("./PlotUtil.jl");

println("\n----Saving Solutions----");

save_solution(res2.minimizer);