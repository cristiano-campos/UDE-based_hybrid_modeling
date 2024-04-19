#A module provides a namespace in which you can define and access
#theses objects, preventing naming conflicts with objects define outside
#of the module
module CommandLineParser 

    using ArgParse
    
    export parseCommandLine2, parseCommandLine

    function parseCommandLine2()
        #define a set of command-line arguments that are expected to be
        #passed to the scriot
        s = ArgParseSettings()
        @add_arg_table s begin 
            "--folder-model"
                help = "Folder in which the model and json experiment config is stored"
                required = true 
            "--config"
                help = "Pass the json file with the experiment config"
                required = true 
            "--increase-train-size-by"
                arg_type = Int 
                default = 1
            "--user-params"
                help = "Paramereters for inference"
                arg_type = String;
                default = "[]";
        end
        return parse_args(s)
    end
    
    function parseCommandLine()
        s = ArgParseSettings()
        @add_arg_table s begin
            "--folder-model"
                help = "folder in witch the model is stored"
                required = true
            "--condition-folder"
                help = "folder in which the conditions are saved in BSON format"
                arg_type = String
                default = "conditions"
            "--results-folder"
                help = "folder in which the results will be saved"
                arg_type = String
                default = "results"
            "--loss-file-name"
                help = "file that has the loss function that must be optimized"
                arg_type = String
                default = "loss_function.jl"
            "--condition-name-prefix"
                help = "the prefix of the file name with the conditions"
                arg_type = String
                default = "condition"
            "--train-size"
                help = "how many conditions should be used to train"
                arg_type = Int
                required = true
            "--test-size"
                help = "how many conditions should be used to test"
                arg_type = Int
                default = 0
            "--n-epochs"
                help = "how many epochs to be run"
                arg_type = Int
                default = 1
            "--ad-maxiters"
                help = "maxiters for ADAM optimizer"
                arg_type = Int
                default = 5000
            "--bs-maxiters"
                help = "maxiters for BFGS optimizer"
                arg_type = Int
                default = 2500
        end
        return parse_args(s)
    end

end 