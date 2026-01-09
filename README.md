# porousflow
master repo 


Permeability generate config command for cpu:
```bash
python3 ./configs/generate_configs.py --mode cpu --models ViT-T16 ViT-S16 ViT-B16 ViT-L16 --exp-name vit_perm --output-dir results --task perm
```
Dispersion:
```bash
python3 ./configs/generate_configs.py --mode cpu --models ConvNeXt-Atto ConvNeXt-V2-Atto ConvNeXt-RMS-Atto --exp-name atto_convnext_disp --output-dir results --task dispersion
```

Execution command for cpu:
```bash

```