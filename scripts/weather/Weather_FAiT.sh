if [ ! -d "./logs/FAiT/ablation/attn" ]; then
    mkdir -p ./logs/FAiT/ablation/attn
fi

# forecasting_Weather_d256_e32_b32_l2_lr5e-4_sf0_wp1_os1_250829_211954_FAiT_custom_sl-96_pl-96_var-21_dm-256_stages-4_2P2i_1P1i_dec-2_des-Exp_0  
# mae: 0.19034, mse: 0.15454, r2: -10.57742, pear: 0.41655, mase: 12.11435

# forecasting_Weather_d128_e512_b32_l2_lr1e-4_sf0_wp1_os1_250830_042122_FAiT_custom_sl-96_pl-192_var-21_dm-128_stages-4_2P2i_1P1i_dec-2_des-Exp_0  
# mae: 0.23965, mse: 0.20283, r2: -1.68351, pear: 0.37007, mase: 15.14635

# forecasting_Weather_d128_e256_b16_l3_lr5e-4_sf0_wp1_os1_250829_045932_FAiT_custom_sl-96_pl-336_var-21_dm-128_stages-4_2P3i_1P1i_dec-2_des-Exp_0  
# mae: 0.28024, mse: 0.26081, r2: -1.23553, pear: 0.33890, mase: 18.19411

# forecasting_Weather_d1024_e64_b16_l3_lr1e-4_sf0_wp1_os1_250829_020700_FAiT_custom_sl-96_pl-720_var-21_dm-1024_stages-4_2P3i_1P1i_dec-2_des-Exp_0  
# mae: 0.33372, mse: 0.34034, r2: -0.90980, pear: 0.30938, mase: 22.02812

model_name=FAiT
seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
train_ratios=(1.0 1.0 1.0 1.0)

d_models=(256 128 128 1024)
embed_size=(32 512 256 64)
bs=(32 32 16 16)
num_layer=(2 2 3 3)

cuda_ids1=(0 0 0 0)
dropout=(0.1 0.1 0.1 0.1)
learning_rate=(5e-4 1e-4 1e-4 5e-4)

lradj=(cosine cosine cosine cosine)

attn_enhance=1
attn_softmax_flag=(0 1)
attn_weight_plus=(0 1)
attn_outside_softmax=(0 1)

for ((i = 0; i < 4; i++))
do
  for ((k1 = 0; k1 < 1; k1++))
  do
    for ((k2 = 1; k2 < 2; k2++))
    do
      for ((k3 = 1; k3 < 2; k3++))
      do

        seq_len=${seq_lens[i]}
        pred_len=${pred_lens[i]}
        train_ratio=${train_ratios[i]}

        export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}

        python -u run.py \
          --is_training 1 \
          --root_path ./dataset/weather/ \
          --data_path weather.csv \
          --model_id Weather_FAiT_${seq_len}_${pred_len}_${lradj[i]}_e${attn_enhance}_s${attn_softmax_flag[k1]}\
_p${attn_weight_plus[k2]}_o${attn_outside_softmax[k3]}_attn_ablation \
          --model $model_name \
          --data custom \
          --features M \
          --seq_len ${seq_len} \
          --pred_len ${pred_len} \
          --enc_in 21 \
          --dec_in 21 \
          --c_out 21 \
          --des 'Exp' \
          --embed_size ${embed_size[i]} \
          --d_model ${d_models[i]} \
          --d_ff ${d_models[i]} \
          --batch_size ${bs[i]} \
          --learning_rate ${learning_rate[i]} \
          --itr 1 \
          --e_layers ${num_layer[i]} \
          --lossfun_alpha 0.5 \
          --test_batch_size 16 \
          --test_mode 0 \
          --CKA_flag 0 \
          --fix_seed 1 \
          --resume_training 0 \
          --save_every_epoch 0 \
          --use_revin 1 \
          --use_norm 1 \
          --send_mail 0 \
          --save_pdf 0 \
          --train_epochs 30 \
          --patience 3 \
          --lradj ${lradj[i]} \
          --loss_mode L1 \
          --train_ratio $train_ratio \
          --dropout ${dropout[i]} \
          --plot_mat_flag 1 \
          --linear_attention 0 \
          --attn_enhance ${attn_enhance} \
          --attn_softmax_flag ${attn_softmax_flag[k1]} \
          --attn_weight_plus ${attn_weight_plus[k2]} \
          --attn_outside_softmax ${attn_outside_softmax[k3]} \
          2>&1 | tee -a logs/FAiT/ablation/attn/FAiT_attn_abl_Weather_${seq_len}_${pred_len}_e${attn_enhance}\
_s${attn_softmax_flag[k1]}_p${attn_weight_plus[k2]}_o${attn_outside_softmax[k3]}.log &\

      done
    done
  done
done




