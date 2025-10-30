# bash code for reproducing our experiments
for project in fed-w8a fed-phishing fed-realsim
do
    for model in logistic SVM
    do
        e=1
        for c in 8 16 32
        do
            python main_FL.py -p $project -seed 444 -fl FedAvg -C $c -E $e --global_epoch 1000 --min_sample 16
            python main_FL.py -p $project -seed 444 -fl FedProx -C $c -E $e --global_epoch 1000 --min_sample 16
            python main_FL.py -p $project -seed 444 -fl MOON -C $c -E $e --global_epoch 1000 --min_sample 16
        done
    done
done
for project in fed-femnist fed-cinic10
do
    for model in DNN Res18 Res50
        for c in 8 16 32
        do
            for e in 1 5 10 20 
            do
                python main_FL.py -p $project -seed $seed -fl FedAvg -C $c -E $e --global_epoch 1000 --min_sample 16
                python main_FL.py -p $project -seed $seed -fl FedProx -C $c -E $e --global_epoch 1000 --min_sample 16
                python main_FL.py -p $project -seed $seed -fl MOON -C $c -E $e --global_epoch 1000 --min_sample 16
            done
        done
    done
done
for project in fed-shakespeare
do
    for model in lstm
        for c in 8 16 32
        do
            for e in 1 5 10 20 
            do
                python main_FL.py -p $project -seed $seed -fl FedAvg -C $c -E $e --global_epoch 1000 --min_sample 16
                python main_FL.py -p $project -seed $seed -fl FedProx -C $c -E $e --global_epoch 1000 --min_sample 16
                python main_FL.py -p $project -seed $seed -fl MOON -C $c -E $e --global_epoch 1000 --min_sample 16
            done
        done
    done
done