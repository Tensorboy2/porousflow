# Predicting Transport Properties of Porous Media Using Machine Learning
By Sigurd Vargdal

This repository contains the code used to produce the results of my master's thesis. While the code is relatively easy to run, my execution was done on several hpc resources. 

## Data generation
The first step in data generation, is generating the domains. The domains are two dimensional porous media, represented by a boolean matrix. It is generated using the periodic version of the **binary_blobs** function from Scikit-image. Each domain is checked for percolation such that the required $24~000$ images all percolate. Some images contains disconnected fluid clusters. These are filled in for numerical efficiency. To execute run:
```bash
python3 src/porousflow/media_generator/media_generator.py N path
```
Where $N$ is the number of samples required and path is the path to save the images. Or with **slurm_generate_media.sh** on slurm clusters.

The thesis runs on two targets, permeability and dispersion coefficients. Permeability is obtained from Lattice-Boltzmann (LBM) simulations in. This is done using the bash script **slurm_run_lbm.sh** for on a cluster with slurm. This launches array jobs of the lbm execution for each domain. 

For dispersion coefficients, the velocity fields of the LBM simulation is used. The dispersion coefficients are obtained through tracer particle simulations using the Euler-Mayorama equation. Is executed using the **slurm_run_dispersion_array.sh**.

## Model training
The model training can be prepared using the *config/generate_configs.py*. The script contains logic for preparing bash and slurm scripts for training on CPU, single GPU, or on a GPU cluster using slurm. No DDP logic is implemented as our models all fit with the dataset in vram. To generate a config the **--task** is set to either permeability or dispersion. Then the device is set to either cpu or gpu using **--device**. If the task is run on a single GPU station without slurm, add the flag **--herbie**. Next the models to train is set using predefined presets **--preset** in the script. The same logic holds for which experiemnts to ru, wheter it is to sweep some hyperparameter or just execute. It must be defined as an "experiment". Lastly a experiment name **--exp-name** to define a distinct name and output folder. Execution looks like this:
```bash
python3 ./configs/generate_configs.py --task _ --device _ (--herbie) --perset _ --experiment _ --exp-name _
```
Then execute the resulting bash or slurm script.

### Zarr->Numpy
If ram on GPU is no issue, it is recommended to convert the zarr objects to numpy for faster dataloader. 

### Testing
Running a model on the test data is done through **run_modle_test.py** using for example:
```bash
python3 run_model_test.py --pretrained_path results/dispersion_all_models_2/ConvNeXt-Atto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth --model 'convnext' --model_name 'ConvNeXt-Atto_asinh' --size 'atto' --version 'v1' --task 'dispersion' --pe_encoder log --loss_function 'mse'
```


## Plotting
Plotting the results can be done with the specific metrics **.zarr** folders. 