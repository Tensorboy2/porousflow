import torch
from src.ml.models.convnext import load_convnext_model
import time
import numpy as np

random_data = torch.randn(100, 128, 1, 128, 128)

models = [
    load_convnext_model({'model_name': 'ConvNeXt', 'version': 'rms', 'size': 'atto'}),
    load_convnext_model({'model_name': 'ConvNeXt', 'version': 'v1',  'size': 'atto'}),
    load_convnext_model({'model_name': 'ConvNeXt', 'version': 'v2',  'size': 'atto'}),
]

model_names = ['ConvNeXt-rms', 'ConvNeXt-v1', 'ConvNeXt-v2']

# Forward pass
print("=== Forward pass ===")
for model, name in zip(models, model_names):
    model.eval()
    batch_times = []
    with torch.no_grad():
        for i in range(100):
            t0 = time.perf_counter()
            output = model(random_data[i])
            batch_times.append(time.perf_counter() - t0)
    times = np.array(batch_times)
    print(f"{name}: {times.mean():.2f} ± {times.std():.2f} s/batch")

# Backward pass
print("\n=== Backward pass ===")
for model, name in zip(models, model_names):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    batch_times = []
    for i in range(100):
        optimizer.zero_grad()
        t0 = time.perf_counter()
        output = model(random_data[i])
        loss = output.mean()
        loss.backward()
        optimizer.step()
        batch_times.append(time.perf_counter() - t0)
    times = np.array(batch_times)
    print(f"{name}: {times.mean():.2f} ± {times.std():.2f} s/batch")