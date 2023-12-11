for i in 'GCN' 'SGC' 'SAGE' 'GIN' 'JKNet' 'MLP'
do
    python -u LargeScaleCondensing.py --dataset ogbn-arxiv --edge_pred aggr --reduction_rate=0.01 --gpu_id=0 --model=${i}
    python -u LargeScaleCondensing_induct.py --dataset reddit --edge_pred aggr --reduction_rate=0.002 --gpu_id=0 --model=${i}
    python -u LargeScaleCondensing_induct.py --dataset deddit2 --edge_pred aggr --reduction_rate=0.002 --gpu_id=0 --model=${i}
done