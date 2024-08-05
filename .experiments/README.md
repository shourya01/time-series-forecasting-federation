## init_model_run.ipynb

Tests out the ComStock dataloader, and different time-series forecasting models from data from the dataloader. Tests out the capability of the model in different data types.

*NOTE:* A lot of files have local filepaths. Also, the dataset is stored locally on my machine. Thus it will be difficult to independently run the script by just cloning this repo.

## 1*.py 

These scripts are meant to run classical FL on a HPC. Uses ComStock dataset (downloaded remotely), multiple time-series forecasting models, and 2 GPUs. Here's the bash script used to generate the `.yaml` files and run the script (note that the script uses gRPC communication).

```
# Load OpenMPI
module load gcc/9.2.0-r4tyw54
module load cuda/11.4.0-gqbcqie
module load openmpi

# activate conda envirnoment
module load anaconda3/2023-01-11
conda activate appfl
conda info

RANDSTR=$(date +%s%N | sha256sum | base64 | head -c 10)
NUM_CLIENTS=12
NUM_STEPS=25

# change directory
cd /home/sbose/time-series-forecasting-federation/.experiments

# list of models
MODELS=("lstm_ar" "darnn" "transformer" "transformer_ar" "logtrans" "informer" "autoformer" "fedformer_fourier" "crossformer" "mlstm")

# loop
for model in "${MODELS[@]}"; do
    # generate configs
    python 1_yaml_generator.py --num_clients $NUM_CLIENTS --model "$model" --num_local_steps $NUM_STEPS --expID "${RANDSTR}"
    # run simultaneous
    for ((i=0; i<NUM_CLIENTS; i++)); do
        if [ "$i" -eq 0 ]; then
            python 1_run_server.py &
        fi
        python 1_run_client.py --client_id $i & 
    done
    wait
done 
```

*NOTE:* A lot of files have local filepaths. Also, the dataset is stored locally on my machine. Thus it will be difficult to independently run the script by just cloning this repo.