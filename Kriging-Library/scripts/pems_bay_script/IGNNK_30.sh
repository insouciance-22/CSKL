#export CUDA_VISIBLE_DEVICES=0

model_name=IGNNK

python -u run.py \
  --model $model_name \
  --is_training 1 \
  --dataset pems_bay \
  --scaler_method fixed \
  --scaler_value 50 \
  --test_name '2024-09-18_11-16-45_%30.npz' \
  --T_test 0.3 \
  --T_val 0.2 \
  --N_test 0.3 \
  --N_val 0.1 \
  --seq_len 16 \
  --ignore_zero 1 \
  --fill_zero knn-mean \
  --d 100 \
  --k_DGCN 1 \
  --tra_s 0.9 \
  --tra_m 0.2 \
  --loss MSE \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --batch_size 4 \
  --num_workers 0 \
  --itr 1 \
  --patience 5 \
  --lradj type0 \
  --use_gpu 1