if [ ! -d "./logs/FAiT/ablation/attn" ]; then
    mkdir -p ./logs/FAiT/ablation/attn
fi

model_name=FAiT
# forecasting_ETTh1_FAiT_96_96_e1_s0_p1_o1_attn_ablation_250823_114332_FAiT_ETTh1_sl-96_pl-96_var-7_dm-1024_stages-4_2P2i_1P1i_dec-2_des-Exp_0  
# mae: 0.38351, mse: 0.36653, r2: -0.07779, pear: 0.57662, mase: 1.87121

# forecasting_ETTh1_FAiT_96_192_e1_s0_p1_o1_attn_ablation_250823_115517_FAiT_ETTh1_sl-96_pl-192_var-7_dm-1024_stages-4_2P2i_1P1i_dec-2_des-Exp_0  
# mae: 0.41932, mse: 0.42336, r2: -0.03423, pear: 0.54014, mase: 2.04964

# forecasting_ETTh1_FAiT_96_336_d512_e8_b4_l1_250823_192431_FAiT_ETTh1_sl-96_pl-336_var-7_dm-512_stages-4_2P1i_1P1i_dec-2_des-Exp_0  
# mae: 0.43765, mse: 0.46247, r2: -0.04514, pear: 0.51855, mase: 2.16234

# forecasting_ETTh1_FAiT_96_720_e1_s0_p1_o1_attn_ablation_250821_230416_FAiT_ETTh1_sl-96_pl-720_var-7_dm-512_stages-4_2P2i_1P1i_dec-2_des-Exp_0  
# mae: 0.46648, mse: 0.46934, r2: 0.01271, pear: 0.48499, mase: 2.27624

seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
train_ratios=(1.0 1.0 1.0 1.0)

d_models=(1024 1024 512 512)  # new
embed_size=(16 16 8 16)
bs=(8 8 4 32)
num_layer=(2 2 1 2)


cuda_ids1=(0 0 0 0)
learning_rate=(1e-4 1e-4 1e-4 1e-4)
dropout=(0.1 0.2 0.2 0.2)  # new

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
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_FAiT_${seq_len}_${pred_len}_e${attn_enhance}_s${attn_softmax_flag}\
_p${attn_weight_plus}_o${attn_outside_softmax}_attn_ablation \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len ${seq_len} \
      --pred_len ${pred_len} \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
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
      --resume_epoch 0 \
      --save_every_epoch 0 \
      --use_revin 1 \
      --use_norm 1 \
      --send_mail 0 \
      --save_pdf 0 \
      --train_epochs 50 \
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
      2>&1 | tee -a logs/FAiT/ablation/attn/FAiT_attn_abl_ETTh1_${seq_len}_${pred_len}_e${attn_enhance}\
_s${attn_softmax_flag}_p${attn_weight_plus}_o${attn_outside_softmax}.log 
done