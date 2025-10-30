# bash code for reproducing our experiments
for project in fed-admm-phishing fed-admm-w8a
do
    for model in logistic SVM
    do
        for c in 8 16 32
        do
            python main_FL.py -p $project -seed 444 -fl FedNew -model $model -C $c -E 1 --global_epoch 1000 --rho 0.1 --alpha 0.25  --min_sample 16

        done
    done
done

for project in fed-ns-phishing fed-ns-w8a
do
    for model in logistic SVM
    do
        for c in 8 16 32 
        do
            for s in 5 15 25 50 100
            do 
                python main_FL.py -p $project -seed 444 -fl FedNS -model $model -C $c -E 1 --global_epoch 1000 --sketch_mu 0.01 --sketch_m $s  --min_sample 16
            done
        done
    done
done