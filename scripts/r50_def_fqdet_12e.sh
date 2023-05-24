#!/bin/bash

exp_name=r50_def_fqdet_12e
out=outputs/$exp_name
mkdir -p $out

nod=2
bat=1
wor=2

cpf=$out/checkpoint.pth
epo=12
lrd=9

arch=bch
core_cfg1=configs/core_in_proj.py
core_cfg2=configs/def_encoder.py
head_cfg=configs/fqdet.py

torchrun --nproc_per_node=$nod main.py --output_dir $out --batch_size $bat --num_workers $wor --checkpoint_full $cpf \
--epochs $epo --lr_drops $lrd --arch_type $arch --resnet_out_ids 3 4 5 --cores cfg cfg \
--core_cfg_paths $core_cfg1 $core_cfg2 --heads cfg --head_cfg_paths $head_cfg
