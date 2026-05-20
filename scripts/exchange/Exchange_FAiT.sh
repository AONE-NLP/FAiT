if [ ! -d "./logs/FAiT/ablation/attn" ]; then
    mkdir -p ./logs/FAiT/ablation/attn
fi

model_name=FAiT

# forecasting_Exchange_d128_e32_b32_l3_250825_122548_FAiT_custom_sl-96_pl-96_var-8_dm-128_stages-4_2P3i_1P1i_dec-2_des-Exp_0  
# mae: 0.19782, mse: 0.08028, r2: -2.87910, pear: 0.01308, mase: 8.03975

# forecasting_Exchange_FAiT_96_192_e1_s0_p1_o1_attn_ablation_250826_115446_FAiT_custom_sl-96_pl-192_var-8_dm-256_stages-4_2P3i_1P1i_dec-2_des-Exp_0  
# mae: 0.29545, mse: 0.17315, r2: -2.37192, pear: -0.08768, mase: 11.67672

# forecasting_Exchange_d512_e8_b32_l3_250827_104303_FAiT_custom_sl-96_pl-336_var-8_dm-512_stages-4_2P3i_1P1i_dec-2_des-Exp_0  
# mae: 0.40760, mse: 0.32118, r2: -2.42616, pear: -0.17927, mase: 16.22970

# forecasting_Exchange_d512_e64_b32_l4_250827_074204_FAiT_custom_sl-96_pl-720_var-8_dm-512_stages-4_2P4i_1P1i_dec-2_des-Exp_0  
# mae: 0.67228, mse: 0.79483, r2: -2.42421, pear: 0.07463, mase: 26.89624

seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
train_ratios=(1.0 1.0 1.0 1.0)

d_models=(128 256 512 512)
embed_size=(32 64 8 64)
bs=(32 8 32 32)
num_layer=(3 3 3 4)

cuda_ids1=(0 0 0 0)

learning_rate=(1e-4 1e-4 1e-4 1e-4)
dropout=(0.0 0.0 0.0 0.0)
train_epochs=(20 20 20 20)

attn_enhance=1
attn_softmax_flag=0
attn_weight_plus=1
attn_outside_softmax=1


for ((i = 0; i < 4; i++))
do

    seq_len=${seq_lens[i]}
    pred_len=${pred_lens[i]}

    export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}

    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/exchange_rate/ \
      --data_path exchange_rate.csv \
      --model_id Exchange_FAiT_${seq_len}_${pred_len}_e${attn_enhance}_s${attn_softmax_flag}\
_p${attn_weight_plus}_o${attn_outside_softmax}_attn_ablation \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len ${seq_len} \
      --pred_len ${pred_len} \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --embed_size ${embed_size[i]} \
      --d_model ${d_models[i]} \
      --d_ff ${d_models[i]} \
      --batch_size  ${bs[i]} \
      --learning_rate ${learning_rate[i]} \
      --itr 1 \
      --e_layers ${num_layer[i]} \
      --lossfun_alpha 0.5 \
      --test_batch_size 16 \
      --test_mode 0 \
      --CKA_flag 0 \
      --fix_seed 1 \
      --resume_training 0 \
      --resume_epoch 0 \
      --save_every_epoch 0 \
      --use_revin 1 \
      --use_norm 1 \
      --send_mail 0 \
      --save_pdf 0 \
      --train_epochs ${train_epochs[i]} \
      --patience 3 \
      --lradj type1 \
      --loss_mode L1 \
      --train_ratio 1.0 \
      --dropout ${dropout[i]} \
      --plot_mat_flag 0 \
      --attn_enhance ${attn_enhance} \
      --attn_softmax_flag ${attn_softmax_flag} \
      --attn_weight_plus ${attn_weight_plus} \
      --attn_outside_softmax ${attn_outside_softmax} \
      2>&1 | tee -a logs/FAiT/ablation/attn/FAiT_attn_abl_Exchange_${seq_len}_${pred_len}_e${attn_enhance}\
_s${attn_softmax_flag}_p${attn_weight_plus}_o${attn_outside_softmax}.log 
done

