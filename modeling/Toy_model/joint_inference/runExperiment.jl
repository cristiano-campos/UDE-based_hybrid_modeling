using Random, JSON, Combinatorics 
 
# Lista com os valores de 1 a 12
initial_params = ["kf1", "kr1", "kcat1", "kf2", "kr2", "kcat2", "kf3", "kr3", "kcat3", "kf4", "kr4", "kcat4"]

# Conjunto para armazenar os vetores de parâmetros passados
past_params = Set{String}()

# Realiza 12 rodadas
#for round in 1:length(initial_params)
#    println("Rodada: ", round)

    # Realiza 5 iterações para cada rodada
#   for i in 1:5
        # Embaralha a lista de parâmetros
#        shuffle!(initial_params)

        # Seleciona os primeiros 'round' parâmetros e ordena
#        current_params = sort(initial_params[1:round])

        # Converte para string para poder ser armazenado no conjunto
#        current_params_str = JSON.json(current_params)

        # Se já vimos esse vetor de parâmetros antes, pula para a próxima iteração
#        if current_params_str in past_params
#            println("Iteração ", i, ": Parâmetros ", current_params, " já utilizados. Pulando para a próxima iteração.")
#            continue
#        end

        # Adiciona ao conjunto de vetores de parâmetros passados
#        push!(past_params, current_params_str)

#        println("Iteração ", i, ": ", current_params)
        # Chama o comando Julia passando os parâmetros
#        run(`julia src/mc_experiment_runner.jl --folder-model model --config experiment.json --user-param $(JSON.json(current_params))`)
#    end
#end

# Calcula o total de combinações possíveis para cada rodada
total_combinations = [length(collect(combinations(initial_params, r))) for r in 1:length(initial_params)]

# Realiza 4 rodadas
for round in 1:length(initial_params)
    println("Rodada: ", round)
    combinations_found = 0

    # Realiza 3 iterações para cada rodada
    for i in 1:4
        found_new_params = false
        attempts = 0

        while !found_new_params
            # Embaralha a lista de parâmetros
            shuffle!(initial_params)

            # Seleciona os primeiros 'round' parâmetros e ordena
            current_params = sort(initial_params[1:round])
            current_params_str = JSON.json(current_params)

            # Verifica se já utilizou este conjunto de parâmetros
            if !(current_params_str in past_params)
                # Marca que encontrou um novo conjunto de parâmetros
                found_new_params = true
                combinations_found += 1

                # Adiciona ao conjunto de vetores de parâmetros passados
                push!(past_params, current_params_str)

                println("Iteração ", i, ": ", current_params)
                # Chama o comando Julia passando os parâmetros
                run(`../julia-1.10.0/bin/julia src/mc_experiment_runner.jl --folder-model model --config experiment.json --user-param $(JSON.json(current_params))`)
            end

            attempts += 1
            # Verifica se todas as combinações possíveis foram tentadas
            if attempts > 100 || combinations_found == total_combinations[round]
                println("Todas as combinações para a rodada $round foram tentadas.")
                break
            end
        end
    end
end
