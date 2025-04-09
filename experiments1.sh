# bash code for reproducing our experiments
for project in fednys-admm-femnist # celeba shakespeare
do
    for seed in 444
    do
        e=1
        for c in 8 16 32 64
        do
            python main_FL.py -p $project -seed $seed -fl FedNew -C $c -E $e --global_epoch 1000 --rho 0.1 --alpha 0.25 --col_opt 1 --min_sample 16
            python main_FL.py -p $project -seed $seed -fl FedNew -C $c -E $e --global_epoch 1000 --rho 0.1 --alpha 0.25 --col_opt 2 --min_sample 16
            python main_FL.py -p $project -seed $seed -fl FedNew -C $c -E $e --global_epoch 1000 --rho 0.1 --alpha 0.25 --col_opt 5 --min_sample 16
            python main_FL.py -p $project -seed $seed -fl FedNew -C $c -E $e --global_epoch 1000 --rho 0.1 --alpha 0.25 --col_opt 10 --min_sample 16
        done
        for c in 8 16 32 64
        do
            python main_FL.py -p $project -seed $seed -fl FedNew -C $c -E $e --global_epoch 1000 --rho 0.1 --alpha 0.25 --col_opt 1 --min_sample 16 --stoch True
            python main_FL.py -p $project -seed $seed -fl FedNew -C $c -E $e --global_epoch 1000 --rho 0.1 --alpha 0.25 --col_opt 2 --min_sample 16 --stoch True
            python main_FL.py -p $project -seed $seed -fl FedNew -C $c -E $e --global_epoch 1000 --rho 0.1 --alpha 0.25 --col_opt 5 --min_sample 16 --stoch True
            python main_FL.py -p $project -seed $seed -fl FedNew -C $c -E $e --global_epoch 1000 --rho 0.1 --alpha 0.25 --col_opt 10 --min_sample 16 --stoch True
        done
    done
done