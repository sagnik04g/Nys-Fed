for project in fed-done-phishing fed-done-w8a
do
    for e in 1 5 10 20 
    do 
        for c in 8 16 32
        do
            for model in logistic SVM
            do
                for done_alpha in 0.001 0.05 0.01 1 5 10
                do
                    python main_FL.py -p $project -seed 444 -fl FedAvg -C $c -E $e --global_epoch 1000 --done_alpha $done_alpha --min_sample 16
                done
            done
        done
    done
done