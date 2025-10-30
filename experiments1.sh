# bash code for reproducing our experiments
for project in fednys-admm-femnist fednys-admm-cinic10
do
    for e in 1 5 10 20 
    do
        for c in 8 16 32
        do
            for model in DNN Res18 Res50
            do
                python main_FL.py -p $project -seed 444 -fl FedNew -C $c -E $e --global_epoch 1000 --rho 0.1 --alpha 0.25 --col_opt 1 --min_sample 16
                python main_FL.py -p $project -seed 444 -fl FedNew -C $c -E $e --global_epoch 1000 --rho 0.1 --alpha 0.25 --col_opt 5 --min_sample 16
                python main_FL.py -p $project -seed 444 -fl FedNew -C $c -E $e --global_epoch 1000 --rho 0.1 --alpha 0.25 --col_opt 10 --min_sample 16
                python main_FL.py -p $project -seed 444 -fl FedNew -C $c -E $e --global_epoch 1000 --rho 0.1 --alpha 0.25 --col_opt 20 --min_sample 16
            done
        done
    done
done