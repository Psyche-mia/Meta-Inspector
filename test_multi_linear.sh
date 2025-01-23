device=0

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the visa dataset
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
        save_dir=./checkpoints/${base_dir}/2024_08_02/
        CUDA_VISIBLE_DEVICES=${device} python test_multi.py --dataset visa \
        --data_path /mnt/IAD_datasets/visa_anomaly_detection/VisA_20220922 --save_path ./results/${base_dir}/2024_08_04 \
        --checkpoint_path ${save_dir}epoch_15.pth \
        --features_list 6 12 18 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
        --subset_size 100
    wait
    done
done


# LOG=${save_dir}"res.log"
# echo ${LOG}
# depth=(9)
# n_ctx=(12)
# t_n_ctx=(4)
# for i in "${!depth[@]}";do
#     for j in "${!n_ctx[@]}";do
#     ## train on the VisA dataset
#         base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
#         save_dir=./checkpoints/${base_dir}/
#         CUDA_VISIBLE_DEVICES=${device} python test.py --dataset visa \
#         --data_path /remote-home/iot_zhouqihang/data/Visa --save_path ./results/${base_dir}/zero_shot \
#         --checkpoint_path ${save_dir}epoch_15.pth \
#         --features_list 6 12 18 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
#     wait
#     done
# done