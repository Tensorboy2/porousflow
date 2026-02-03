from pathlib import Path
import zarr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


Pes = [0.1,10,50,100,500]
colors = plt.get_cmap("tab10").colors

plt.figure(figsize=(6,6))
for i in range(5):
    c = colors[i % len(colors)]
    # results/pe_sweep_2/ResNet-152_lr-0.0005_wd-0.01_bs-128_epochs-400_cosine_warmup-625.0_clipgrad-True_pe-encoder-None_pe-0_rmse_metrics.zarr
    data = zarr.open(f'results/pe_sweep_2/ResNet-152_lr-0.0005_wd-0.01_bs-128_epochs-400_cosine_warmup-625.0_clipgrad-True_pe-encoder-None_pe-{i}_rmse_metrics.zarr', mode='r')
    # train_loss = data['train_loss'][:]
    # val_loss = data['val_loss'][:]
    R2_train = data['R2_train'][:]
    R2_val = data['R2_val'][:]

    plt.plot(1-R2_val, color=c, label=f'Pe: {Pes[i]}')
    plt.plot(1-R2_train,'--',color=c,alpha=0.3)

style_legend = [
Line2D([0], [0], color='black', lw=2, linestyle='-', label='Validation'),
Line2D([0], [0], color='black', lw=2, linestyle='--', label='Train'),
]
plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + style_legend)

plt.yscale('log')
# plt.xscale('log')
plt.xlabel('Epochs')
plt.ylabel(r'$1-R^2$')
plt.title("ResNet-152 Pe-sweep with archsinh scaling")
plt.grid(alpha=0.3)
plt.savefig('resnet_pe_sweep_r2.pdf')
plt.close()

# plt.figure(figsize=(6,6))
# for j in range(5):
#     c = colors[j % len(colors)]

#     data = zarr.open(f'results/pe_sweep/ConvNeXt-Atto_lr-0.0008_wd-0.1_bs-128_epochs-200_cosine_warmup-2000_clipgrad-True_pe-encoder-_pe-{j}_metrics.zarr', mode='r')
#     train_loss = data['train_loss'][:]
#     val_loss = data['val_loss'][:]
#     # R2_train = data['R2_train'][:]
#     # R2_val = data['R2_val'][:]

#     plt.plot(val_loss, color=c, label=f'Pe: {Pes[i]}')
#     plt.plot(train_loss,'--',color=c,alpha=0.3)

# style_legend = [
# Line2D([0], [0], color='black', lw=2, linestyle='-', label='Validation'),
# Line2D([0], [0], color='black', lw=2, linestyle='--', label='Train'),
# ]
# plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + style_legend)

# plt.yscale('log')
# # plt.xscale('log')
# plt.xlabel('Epochs')
# plt.ylabel(r'$MSE$')
# plt.title("ConvNeXt-Atto Pe-sweep with archsinh scaling")
# plt.grid(alpha=0.3)
# plt.savefig('update_plots/convnext-atto_pe_sweep_mse.pdf')
# plt.close()


# families = {
#     "ConvNeXt-V2": ["Atto", "Femto", "Pico", "Nano", "Tiny", "Small", "Base", "Large"],
#     "ConvNeXt-RMS": ["Atto", "Femto", "Pico", "Nano", "Tiny", "Small", "Base", "Large"],
#     "ConvNeXt": ["Atto", "Femto", "Pico", "Nano", "Tiny", "Small", "Base", "Large"],
#     "ResNet": ["18", "34", "50", "101", "152"],
#     "ViT": ["T16", "S16", "B16", "L16"],
# }
# sizes = {
#     "ConvNeXt-V2": [3388604, 4849684, 8555204, 14985844, 27871588, 49561444, 87708804, 196443844],
#     "ConvNeXt-RMS": [3371724, 4829428, 8528196, 14946324, 27811204, 49438852, 87545348, 196198660],
#     "ConvNeXt": [3373884, 4832020, 8531652, 14951284, 27818596, 49453156, 87564420, 196227268],
#     "ResNet": [11172292, 21280452, 23509956, 42502084, 58145732],
#     "ViT": [5401156, 21419140, 85305604, 302644228],
# }

# def parse_model(path):
#     name = path.stem.split("_lr-")[0] # ConvNeXt-V2-Femto
#     for family in families:
#         if name.startswith(family):
#             variant = name[len(family)+1:] # skip "Family-"
#             return family, variant
#     return None, None

# def get_paramcount(family, variant):
#     idx = families[family].index(variant)
#     return sizes[family][idx]
        
# colormaps = {
#     "ConvNeXt-V2": "plasma",
#     "ConvNeXt-RMS": "cividis",
#     "ConvNeXt": "viridis",
#     "ResNet": "autumn",
#     "ViT": "winter",
# }

# runs = {
#     'permeability_over_parameter_count': 'results/all_models_permeability',
#     'dispersion_pe:_0.1_over_parameter_count':'results/zero_pecle_all_models', 
# }
# tab10 = plt.get_cmap("tab10").colors


# family_colors = {
# family: tab10[i]
# for i, family in enumerate(families.keys())
# }
# markers = ["o", "s", "^", "D", "v"] # circle, square, triangle_up, diamond, triangle_down
# family_markers = {
#     family: markers[i]
#     for i, family in enumerate(families.keys())
# }
# # typical path: results/all_models_permeability/ConvNeXt-V2-Femto_lr-0.0008_wd-0.1_bs-128_epochs-500_cosine_warmup-1000_metrics.zarr
# for k,v in runs.items():
#     files = Path(v).glob("*.zarr")
#     plt.figure(figsize=(4.5,4.5))

#     for file in files:
#         family, variant = parse_model(file)
#         if family is None:
#             continue # skip unknown
#         # c = plt.get_cmap(colormaps[family]).colors[variant]
#         # idx = families[family].index(variant)
#         # n = len(families[family])
#         # cmap = plt.get_cmap(colormaps[family])
#         # c = cmap(idx / (n - 1))
#         c = family_colors[family]
#         m = family_markers[family]
#         paramcount = get_paramcount(family, variant)
#         print(file.name, family, variant, paramcount)
#         data = zarr.open(file, mode='r')
#         R2_val = data['R2_val'][:]
#         r2_max = R2_val.max()
#         plt.plot(paramcount, 1-r2_max, marker=m, color=c,markerfacecolor='none', linestyle='None', markersize=12)

#     for family in families.keys():
#         plt.plot([], [], marker=family_markers[family],
#         color=family_colors[family],
#         linestyle='None',
#         markersize=12,markerfacecolor='none',
#         label=family)
#     plt.legend()

#     plt.ylabel(r'$1-R^2$')
#     plt.yscale('log') 
#     plt.xscale('log') 
#     plt.xlabel('Parameter Count')
#     plt.title(k)
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(f'update_plots/{k}.pdf')
#     plt.close()



# from pathlib import Path
# import zarr
# import matplotlib.pyplot as plt


# pe_encoders = ['straight','log','vector']
# models = ['ConvNeXt-Atto', 'ConvNeXt-Tiny','ConvNeXt-Base']

# fig, ax = plt.subplots(1, 3, figsize=(9, 4.5))

# # colors for pe-encoders
# colors = {"straight":"C0", "log":"C1", "vector":"C2"}

# for i, m in enumerate(models):
#     plotted_encoders = set()  # reset per subplot
#     for encoder in pe_encoders:
#         path = f'results/pe_encoder_sweep_convnext/{m}_lr-0.0008_wd-0.01_bs-128_epochs-200_cosine_warmup-1000_clipgrad-True_pe-encoder-{encoder}_metrics.zarr'

#         # open data
#         data = zarr.open(path, mode='r')
#         R2_val = data['R2_val'][:]
#         R2_train = data['R2_train'][:]

#         # add label only once per encoder
#         label = encoder if encoder not in plotted_encoders else None
#         if label is not None:
#             plotted_encoders.add(encoder)

#         # plot lines
#         ax[i].plot(1-R2_val, color=colors[encoder], label=label)
#         ax[i].plot(1-R2_train, color=colors[encoder], alpha=0.3)

#     # configure subplot
#     ax[i].set_xlabel("Epochs")
#     ax[i].set_ylabel(r"$1-R^2$")
#     ax[i].set_yscale('log')
#     ax[i].set_title(m)
#     ax[i].grid(alpha=0.3)
#     ax[i].legend(title="PE Encoder")

# plt.tight_layout()
# plt.savefig("update_plots/pe_encoder_sweep_convnext_r2.pdf")
# plt.close()
