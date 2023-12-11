# synthetic graph
for nlayers in 2 3 4
do
    for hidden in 128 256 512
    do
        for dropout in 0 0.3 0.5
        do
            for activation in 'sigmoid' 'tanh' 'relu' 'softplus' 'leakyreLU' 'elu'
            do
                python -u LargeScaleCondensing.py --dataset ogbn-arxiv --edge_pred aggr --reduction_rate=0.01 --gpu_id=0 --model=GCN  --nlayers=${nlayers} --hidden=${hidden} --dropout=${dropout} --activation=${activation} --seed=1 
            done
        done
    done
done

for nlayers in 2 3 4
do
    for hidden in 128 256 512
    do
        for dropout in 0 0.3 0.5
        do
            for activation in 'sigmoid' 'tanh' 'relu' 'softplus' 'leakyreLU' 'elu'
            do
                python -u LargeScaleCondensing_induct.py --dataset reddit --edge_pred aggr --reduction_rate=0.002 --gpu_id=0 --model=GCN  --nlayers=${nlayers} --hidden=${hidden} --dropout=${dropout} --activation=${activation} --seed=1 
            done
        done
    done
done

