

python3 run_model_test.py --pretrained_path results/epoch_sweep_all_models/ConvNeXt-Atto_lr-0.0005_wd-0.1_bs-128_epochs-1000_cosine_warmup-0_clipgrad-True_pe-encoder-None_pe-None_mse.pth --model 'convnext' --model_name 'ConvNext-Atto' --size 'atto' --version 'v1' --task 'permeability' --loss_function 'mse'

python3 run_model_test.py --pretrained_path results/epoch_sweep_all_models/ConvNeXt-Small_lr-0.0005_wd-0.1_bs-128_epochs-700_cosine_warmup-0_clipgrad-True_pe-encoder-None_pe-None_mse.pth --model 'convnext' --model_name 'ConvNext-Small' --size 'small' --version 'v1' --task 'permeability' --loss_function 'mse'