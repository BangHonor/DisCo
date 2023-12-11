
for i in  1 2 3 4 5
do
    for r in 0.25 0.5 1.0
    do
        python -u LargeScaleCondensing.py --dataset cora --edge_pred aggr --condensing_loop 1500  --reduction_rate=${r} --gpu_id=0 --model=GCN --seed=${i}
    done
done

for i in  1 2 3 4 5
do
    for r in 0.001 0.005 0.01
    do
        python -u LargeScaleCondensing.py --dataset ogbn-arxiv --edge_pred aggr --condensing_loop 1500 --reduction_rate=${r} --gpu_id=0 --model=GCN --seed=${i}
    done
done

for i in  1 2 3 4 5
do
    for r in 0.005 0.01 0.1
    do
        python -u LargeScaleCondensing.py --dataset ogbn-products --edge_pred aggr --condensing_loop 2500 --reduction_rate=${r} --gpu_id=0 --model=GCN --seed=${i}
    done
done

for i in 1 2 3 4 5
do
    for r in 0.0005 0.001 0.002
    do
        python -u LargeScaleCondensing_induct.py --dataset reddit --edge_pred aggr --condensing_loop 2500 --reduction_rate=${r} --gpu_id=0 --model=GCN --seed=${i}
    done
done

for i in 1 2 3 4 5
do
    for r in 0.0005 0.001 0.002
    do
        python -u LargeScaleCondensing_induct.py --dataset reddit2 --edge_pred aggr --condensing_loop 2500 --reduction_rate=${r} --gpu_id=0 --model=GCN --seed=${i}
    done
done