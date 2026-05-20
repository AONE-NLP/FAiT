if [ ! -d "./logs/FAiT/ablation/attn" ]; then
    mkdir -p ./logs/FAiT/ablation/attn
fi

#export CUDA_VISIBLE_DEVICES=0
# forecasting_ECL_FAiT_96_96_d512_e64_b8_l2_250829_062757_FAiT_custom_sl-96_pl-96_var-321_dm-512_stages-4_2P2i_1P1i_dec-2_des-Exp_0  
# mae: 0.22457, mse: 0.13725, r2: 0.59111, pear: 0.91203, mase: 0.91872

# forecasting_ECL_FAiT_96_192_d256_e32_b8_l4_250828_001502_FAiT_custom_sl-96_pl-192_var-321_dm-256_stages-4_2P4i_1P1i_dec-2_des-Exp_0  
# mae: 0.24326, mse: 0.15549, r2: 0.59631, pear: 0.90466, mase: 1.00326

# forecasting_ECL_FAiT_96_336_d512_e128_b16_l3_250903_231329_FAiT_custom_sl-96_pl-336_var-321_dm-512_stages-4_2P3i_1P1i_dec-2_des-Exp_0  
# mae: 0.25945, mse: 0.17047, r2: 0.68724, pear: 0.89663, mase: 1.08340

# forecasting_ECL_FAiT_96_720_d1024_e8_b32_l3_250901_171049_FAiT_custom_sl-96_pl-720_var-321_dm-1024_stages-4_2P3i_1P1i_dec-2_des-Exp_0  
# mae: 0.28215, mse: 0.19504, r2: 0.69362, pear: 0.88382, mase: 1.18756

model_name=FAiT

seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
train_ratios=(1.0 1.0 1.0 1.0)

d_models=(512 256 512 1024)
embed_size=(64 32 128 8)
bs=(8 8 16 32)
num_layer=(2 4 3 3)

cuda_ids1=(0 0 0 0)
epochs=(50 50 50 50)
lradj=(cosine cosine cosine cosine)

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
      --root_path ./dataset/electricity/ \
      --data_path electricity.csv \
      --model_id ECL_FAiT_${seq_len}_${pred_len}_e${attn_enhance}_s${attn_softmax_flag}\
_p${attn_weight_plus}_o${attn_outside_softmax}_${lradj[i]}_attn_ablation \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len ${seq_len} \
      --pred_len ${pred_len} \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --embed_size ${embed_size[i]} \
      --d_model ${d_models[i]} \
      --d_ff ${d_models[i]} \
      --batch_size ${bs[i]} \
      --learning_rate 5e-4 \
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
      --train_epochs ${epochs[i]} \
      --patience 3 \
      --lradj ${lradj[i]} \
      --loss_mode L1 \
      --train_ratio 1.0 \
      --dropout 0.0 \
      --plot_mat_flag 0 \
      --plot_mat_label ECL_attn011 \
      --attn_enhance ${attn_enhance} \
      --attn_softmax_flag ${attn_softmax_flag} \
      --attn_weight_plus ${attn_weight_plus} \
      --attn_outside_softmax ${attn_outside_softmax} \
      2>&1 | tee -a logs/FAiT/ablation/attn/FAiT_attn_abl_ECL_${seq_len}_${pred_len}_${lradj[i]}_e${attn_enhance}\
_s${attn_softmax_flag}_p${attn_weight_plus}_o${attn_outside_softmax}.log

done