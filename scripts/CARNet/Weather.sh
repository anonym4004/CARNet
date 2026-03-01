
model_name=cSofts
root_path_name=C:/Users/Awsftausif/Desktop/S-Mamba_datasets/weather/
data_path_name=weather.csv
model_id_name=weather
data_name=custom
random_seed=2024

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_96_96' \
  --model $model_name \
  --data $data_name \
  --features M \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 21 \
  --n_vars 21 \
  --d_model 512 \
  --batch_size 16 \
  --d_ff 512 \
  --d_core 128 \
  --learning_rate 0.0001 \
  --cycle 144 \
  --des 'Exp' \
  --freq h \
  --itr 0

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_96_192' \
  --model $model_name \
  --data $data_name \
  --features M \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 1 \
  --enc_in 21 \
  --n_vars 21 \
  --d_model 512 \
  --batch_size 16 \
  --d_ff 512 \
  --d_core 144 \
  --learning_rate 0.0001 \
  --cycle 144 \
  --des 'Exp' \
  --freq t \
  --itr 0

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_96_336' \
  --model $model_name \
  --data $data_name \
  --features M \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 1 \
  --enc_in 21 \
  --n_vars 21 \
  --d_model 512 \
  --batch_size 16 \
  --d_ff 512 \
  --d_core 128 \
  --learning_rate 0.0001 \
  --cycle 144 \
  --des 'Exp' \
  --freq t \
  --itr 0

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_96_720' \
  --model $model_name \
  --data $data_name \
  --features M \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720\
  --e_layers 1 \
  --enc_in 21 \
  --n_vars 21 \
  --d_model 512 \
  --batch_size 16 \
  --d_ff 512 \
  --d_core 128 \
  --learning_rate 0.0001 \
  --cycle 144 \
  --des 'Exp' \
  --freq t \
  --itr 1