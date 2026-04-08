
# python3 run_model_test.py --pretrained_path results/epoch_sweep_all_models/ResNet-18_lr-0.0005_wd-0.1_bs-128_epochs-1000_cosine_warmup-0_clipgrad-True_pe-encoder-None_pe-None_mse.pth --model 'resnet' --model_name 'ResNet-18' --size '18' --version None --task 'permeability' --loss_function 'mse'

# python3 run_model_test.py --pretrained_path results/epoch_sweep_all_models/ConvNeXt-Large_lr-0.0005_wd-0.1_bs-128_epochs-1000_cosine_warmup-0_clipgrad-True_pe-encoder-None_pe-None_mse.pth --model 'convnext' --model_name 'ConvNext-Large' --size 'large' --version 'v1' --task 'permeability' --loss_function 'mse'

# python3 run_model_test.py --pretrained_path results/epoch_sweep_all_models/ConvNeXt-Small_lr-0.0005_wd-0.1_bs-128_epochs-1500_cosine_warmup-3750.0_clipgrad-True_pe-encoder-None_pe-None_mse.pth --model 'convnext' --model_name 'ConvNext-Small' --size 'small' --version 'v1' --task 'permeability' --loss_function 'mse'

# python3 run_model_test.py --pretrained_path results/dispersion_lr_wd_sweep/ConvNeXt-Atto_lr-0.001_wd-0.05_bs-128_epochs-500_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_rmse.pth --model 'convnext' --model_name 'ConvNext-Atto' --size 'atto' --version 'v1' --task 'dispersion' --pe_encoder log --loss_function 'rmse'
python3 run_model_test.py --pretrained_path results/dispersion_all_models_2/ConvNeXt-RMS-Pico_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth --model 'convnext' --model_name 'ConvNext-Pico' --size 'pico' --version 'rms' --task 'dispersion' --pe_encoder log --loss_function 'rmse'

# python3 run_model_test.py --pretrained_path results/dispersion_all_models/ResNet-18_lr-0.005_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth --model 'resnet' --model_name 'ResNet-18' --size '18' --version None --task 'dispersion' --pe_encoder log --loss_function 'mse'

# python3 run_model_test.py --pretrained_path results/dispersion_epoch_sweep/ConvNeXt-Atto_lr-0.005_wd-0.01_bs-128_epochs-600_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth --model 'convnext' --model_name 'ConvNext-Atto' --size 'atto' --version 'v1' --task 'dispersion' --pe_encoder log --loss_function 'mse'
# python3 run_model_test.py --pretrained_path results/epoch_sweep_all_models/ConvNeXt-Atto_lr-0.0005_wd-0.1_bs-128_epochs-1000_cosine_warmup-0_clipgrad-True_pe-encoder-None_pe-None_mse.pth --model 'convnext' --model_name 'ConvNext-Atto' --size 'atto' --version 'v1' --task 'permeability' --loss_function 'mse'
