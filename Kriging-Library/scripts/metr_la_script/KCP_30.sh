#export CUDA_VISIBLE_DEVICES=0

model_name=KCP

python -u run.py \
  --model $model_name \
  --is_training 1 \
  --dataset metr_la \
  --scaler_method standard \
  --test_name '2024-09-06_21-41-39_%30.npz' \
  --T_test 0.3 \
  --T_val 0.2 \
  --N_test 0.3 \
  --N_val 0.1 \
  --seq_len 16 \
  --ignore_zero 1 \
  --fill_zero 'knn-mean' \
  --emb_size 64 \
  --dec_hid 32 \
  --nmb_p 5 \
  --tra_aug 0.2 \
  --rep_num 2 \
  --self_learning_epoch 50 \
  --self_learning_lr 0.001 \
  --downtask_hid 64 \
  --downtask_lr 0.005 \
  --finetune_lr 0.001 \
  --pre_training 1 \
  --load_path '' \
  --loss MSE \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --batch_size 4 \
  --num_workers 0 \
  --itr 1 \
  --patience 5 \
  --lradj type0 \
  --use_gpu 1