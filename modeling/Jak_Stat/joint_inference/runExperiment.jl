using Random, JSON
 
initial_params = ["k50", "k51" , "k58", "k59","k53", "k61"];

# Conjunto para armazenar os vetores de parâmetros passados
past_params = Set{String}()

# Realiza 12 rodadas
for round in 1:length(initial_params)
    println("Rodada: ", round)

    # Realiza 5 iterações para cada rodada
   for i in 1:6
        # Embaralha a lista de parâmetros
        shuffle!(initial_params)

        # Seleciona os primeiros 'round' parâmetros e ordena
        current_params = sort(initial_params[1:round])

        # Converte para string para poder ser armazenado no conjunto
        current_params_str = JSON.json(current_params)

        # Se já vimos esse vetor de parâmetros antes, pula para a próxima iteração
        if current_params_str in past_params
            println("Iteração ", i, ": Parâmetros ", current_params, " já utilizados. Pulando para a próxima iteração.")
            continue
        end

        # Adiciona ao conjunto de vetores de parâmetros passados
        push!(past_params, current_params_str)

        println("Iteração ", i, ": ", current_params)
        # Chama o comando Julia passando os parâmetros
        run(`C:/Users/crist/AppData/Local/Programs/Julia-1.10.0/bin/julia.exe src/mc_experiment_runner.jl --folder-model model --config experiment.json --user-param $(JSON.json(current_params))`)
    end
end
