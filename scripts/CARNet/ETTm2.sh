
model_name=carnet
root_path_name=C:/Users/Awsftausif/Desktop/S-Mamba_datasets/ETT-small//
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
random_seed=2024

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_96_96' \
  --model $model_name \
  --data $data_name \
  --features M \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --n_vars 7 \
  --d_model 128 \
  --batch_size 32 \
  --d_ff 128 \
  --d_core 128 \
  --learning_rate 0.0001 \
  --cycle 96 \
  --des 'Exp' \
  --freq t \
  --train_epochs 7 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_96_192' \
  --model $model_name \
  --data $data_name \
  --features M \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --n_vars 7 \
  --d_model 128 \
  --batch_size 32 \
  --d_ff 128 \
  --d_core 128 \
  --learning_rate 0.0001 \
  --cycle 96 \
  --des 'Exp' \
  --freq t \
  --train_epochs 7 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_96_336' \
  --model $model_name \
  --data $data_name \
  --features M \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 7 \
  --n_vars 7 \
  --d_model 512 \
  --batch_size 32 \
  --d_ff 512 \
  --d_core 128 \
  --learning_rate 0.0001 \
  --cycle 96 \
  --des 'Exp' \
  --freq t \
  --train_epochs 7 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_96_720' \
  --model $model_name \
  --data $data_name \
  --features M \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720\
  --e_layers 2 \
  --enc_in 7 \
  --n_vars 7 \
  --d_model 128 \
  --batch_size 32 \
  --d_ff 128 \
  --d_core 128 \
  --learning_rate 0.0001 \
  --cycle 96 \
  --des 'Exp' \
  --freq t \
  --train_epochs 7 \
  --itr 1