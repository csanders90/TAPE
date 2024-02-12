
from datetime import datetime as dt
import argparse
import uuid
import torch
import wandb

TF_ENABLE_ONEDNN_OPTS=0

##################################################################################################################

# Necessary changes before run this .py to tune hyper-parameters:

# 1. change args.data with desired dataset
#     e.g. datasets/RandomNoise_CDonly_new/training_data_5km.hdf5
# 2. change args.csv_file_name
#     e.g. tune on dataset Realistic_data, default='SCINet_HyParamTune_Realistic_records.csv'
# 3. change args.tuned_hyperparmas for tuned hyperparams and scale
#     e.g. tune {'levels': [2,3,4],'window_size': [32,64,128,256,512],}

##################################################################################################################

def get_args():


    args = parser.parse_args()

    if not args.long_term_forecast:
        args.concat_len = args.window_size - args.horizon

    return args

def get_model(args):

    Exp=Exp_kiglis
    exp=Exp(args)

    return exp

def Hyper_param_Tune():

    torch.manual_seed(4321)  # reproducible
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
    torch.backends.cudnn.enabled = True

    # initialize wandb
    args.log_id = f"{dt.now().strftime('%d_%h_%H_%M')}-{args.model_name}-{args.dataset_name}"
    run = wandb.init(name=args.log_id, config=args)

    # save the files defined in wandb_record_files
    if args.save_files:
        run.log_code(include_fn=wandb_record_files)

    # hyper-params to be swept & change name of run and log_id
    args.levels = wandb.config.levels
    args.window_size = wandb.config.window_size
    run.name = run.name+f"-l{args.levels}-wl{args.window_size}-{uuid.uuid4()}"
    args.log_id = args.log_id+f"-l{args.levels}-wl{args.window_size}-{uuid.uuid4()}"

    # create and watch model
    exp = get_model(args)

    before_train = dt.now().timestamp()
    print("===================Normal-Start=========================")
    normalize_statistic = exp.train()
    after_train = dt.now().timestamp()
    print(f'Training took {(after_train - before_train) / 60} minutes')
    print("===================Normal-End=========================")

if __name__ == '__main__':

    args = get_args()
    args.dataset_name = get_dataset_name(args.data)

    ### prepare config
    # load the tunued hyper-params according to following form:
    #  parameters={
    #               'levels': [2,3,4],
    #               'window_size': [32,64,128,256,512],
    #               }
    sweep_name = args.dataset_name + f': '
    param_dict = {}
    tune_count = 1
    for Name, Scale in args.tuned_hyperparmas.items():
        param_dict[Name] = {'values': Scale}
        sweep_name = sweep_name + f'{Name},'
        tune_count *= len(Scale)
    print(f'sweep name will be: {sweep_name}')
    print(f'how many times will be tuned: {tune_count}')


    ### customized hyper-params tuning
    sweep_configuration = {
    'method': 'grid',
    'name': sweep_name,
    'metric': {'goal': 'minimize', 'name': 'test/ber'},
    'parameters': param_dict
    }
    
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.project, entity=args.entity)
    wandb.agent(sweep_id, function=Hyper_param_Tune, count=tune_count)