# for i in 1 
# do
#     for r in 0.005 0.01 0.02 0.05 0.1
#     do
#         python -u LargeScaleCondensing_Sampled.py --dataset ogbn-papers100M --edge_pred aggr --condensing_loop 2500 --inference True  --reduction_rate=${r} --gpu_id=0 --model=SGC --seed=${i}
#     done
# done

for i in 1 
do
    for r in 0.005 0.01 0.02 0.05 
    do
        python -u LargeScaleCondensing_random.py --dataset ogbn-papers100M --inference True   --method herding --reduction_rate=${r} --gpu_id=1 --model=SGC --seed=${i}
        python -u LargeScaleCondensing_random.py --dataset ogbn-papers100M --inference True   --method kcenter --reduction_rate=${r} --gpu_id=1 --model=SGC --seed=${i}
    
    done
done