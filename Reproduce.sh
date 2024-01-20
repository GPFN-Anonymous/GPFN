#! /bin/sh
#
# This script is to reproduce our results in Table 3.
RPMAX=20

list=("GIN" "MLP" "GCN" "GAT" "GPRGNN" "APPNP")
filter_list=("empty" "log" "Katz" "AGE" "RES" "SGC" "Bernet" "scale-1" "scale-2" "scale-3")

for data in "${data_list[@]}"
do
for filter in "${filter_list[@]}"
do
        for item in "${list[@]}"
        do
        CUDA_VISIBLE_DEVICES=0 python train_model.py --RPMAX $RPMAX \
                --net $item \
                --train_rate 0.025 \
                --val_rate 0.025 \
                --dataset $data \
                --lr 0.01 \
                --alpha 0.1 \
                --missing-rate 0 \
                --filter $filter
        done

        for item in "${list[@]}"
        do
        CUDA_VISIBLE_DEVICES=0 python train_model.py --RPMAX $RPMAX \
                --net $item \
                --train_rate 0.025 \
                --val_rate 0.025 \
                --dataset $data \
                --lr 0.01 \
                --alpha 0.1 \
                --missing-rate 0.3 \
                --filter $filter
        done

        for item in "${list[@]}"
        do
        CUDA_VISIBLE_DEVICES=0 python train_model.py --RPMAX $RPMAX \
                --net $item \
                --train_rate 0.025 \
                --val_rate 0.025 \
                --dataset $data \
                --lr 0.01 \
                --alpha 0.1 \
                --missing-rate 0.6 \
                --filter $filter
        done

        for item in "${list[@]}"
        do
        CUDA_VISIBLE_DEVICES=0 python train_model.py --RPMAX $RPMAX \
                --net $item \
                --train_rate 0.025 \
                --val_rate 0.025 \
                --dataset $data \
                --lr 0.01 \
                --alpha 0.1 \
                --missing-rate 0.9 \
                --filter $filter
        done
done
done
