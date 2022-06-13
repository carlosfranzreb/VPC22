""" Script to run a HiFi-GAN model with a single input for explorative purposes.
Pick the model and the input files below. You can find the inputs in the given
folders."""


import sys

import torch

from core_scripts.data_io.customize_dataset import NII_MergeDataSetLoader
from core_scripts.data_io.io_tools import f_read_raw_mat
import core_scripts.other_tools.list_tools as nii_list_tool
from projects.joint_tts_hifigan import model as original_model
from projects.joint_tts_nsf_hifigan import model as nsf_model
from projects.joint_tts_hifigan.main import main as original_main
from projects.joint_tts_nsf_hifigan.main import main as nsf_main


class Config:
    def __init__(self):
        self.wav_samp_rate = 16000
        self.input_reso = [160, 160, 160]


def run_main(model):
    """Run the main function of the model."""
    sys.argv.extend(["--lr", "0.0001"])
    sys.argv.extend(["--epochs", "1"])
    sys.argv.extend(["--no-best-epochs", "1"])
    sys.argv.extend(["--batch-size", "1"])
    sys.argv.extend(["--num-workers", "1"])
    sys.argv.extend(["--module-config", f"projects.{model}.config_debug"])
    sys.argv.extend(["--module-model", f"projects.{model}.model"])
    if model == "joint_tts_hifigan":
        original_main()
    elif model == "joint_tts_nsf_hifigan":
        nsf_main()
    else:
        raise RuntimeError("Wrong model")


def debug_model(model, dataset):
    """Run the model with the given dataset. WIP (not working yet)"""
    dataset_loader = get_NII_MergeDataSetLoader(dataset)
    loader = dataset_loader.get_loader()
    args = None
    config = Config()
    if model == "joint_tts_hifigan":
        model = original_model
    elif model == "joint_tts_nsf_hifigan":
        model = nsf_model
    else:
        raise RuntimeError("Wrong model")
    generator = model.ModelGenerator(
        dataset_loader.get_in_dim(),
        dataset_loader.get_out_dim(),
        args,
        config,
        dataset_loader.get_data_mean_std(),
    )
    generator.float()
    for data_in, data_tar, data_info, idx_orig in loader:
        out = generator(data_in)
        print(out)


def get_NII_MergeDataSetLoader(dataset):
    """Run the class for debugging."""
    data_dir = f"baseline/exp/am_nsf_data/{dataset}"
    params = {
        "batch_size": 1,
        "shuffle": False,
    }
    dataset_loader = NII_MergeDataSetLoader(
        dataset,  # dataset_name
        nii_list_tool.read_list_from_text(
            f"{data_dir}/scp/data.lst"
        ),  #  list of filenames
        [
            [f"{data_dir}/ppg", f"{data_dir}/xvector", f"{data_dir}/f0"]
        ],  #  list of lists of dirs for input features
        [".ppg", ".xvector", ".f0"],  # input feature name extentions
        [256, 512, 1],  # list of input feature dimensions
        [160, 160, 160],  # list of input feature temporal resolution
        [True, True, True],  # whether to normalize input features
        [[]],  # list of lists of dirs for output features
        [],  # list of output feature name extentions
        [],  # list of output feature dimensions
        [],  # list of output feature temporal resolution
        [],  # whether to normalize target feature
        "./",  # path to the directory of statistics(mean/std)
        "<f4",  # method to load the data
        params=params,  # parameters for torch.utils.data.DataLoader
        truncate_seq=None,  # truncate data sequence into smaller chuncks
        min_seq_len=None,  # minimum length of an utterance
        save_mean_std=False,  # save mean and std
        wav_samp_rate=16000,  # sampling rate of the waveform
        way_to_merge="concatenate",  # concatenate or merge from corpora
        global_arg=None,  # argument parser
        dset_config=None,  # data set configuration
        input_augment_funcs=None,  # input data transformations
        output_augment_funcs=None,  # output data transformations
        inoutput_augment_func=None,  # single data augmentation function
    )
    return dataset_loader


def build_input(dataset, utterance, arr):
    """Input data is the concatenation of ppg, xvector and f0, in that order."""
    data_dir = f"baseline/exp/am_nsf_data/{dataset}"
    ppg = torch.Tensor(f_read_raw_mat(f"{data_dir}/ppg/{utterance}.ppg", 256))
    xvector = torch.Tensor(
        f_read_raw_mat(f"{data_dir}/xvector/{utterance}.xvector", 512)
    )
    f0 = torch.Tensor(f_read_raw_mat(f"{data_dir}/f0/{utterance}.f0", arr.shape[1]))
    all = torch.zeros(arr.shape).squeeze()
    all[:, :256] = ppg  # ppg contains 256 dims per time unit
    all[:, 256:768] = xvector  # duplicate xvec for each time unit
    all[:, 768] = f0  # one freq value per time unit
    return all


def assert_input(dataset, utterance):
    """Assert that the built input equals the input ouput by the dataloader."""
    dataset_loader = get_NII_MergeDataSetLoader(dataset)
    loader = dataset_loader.get_loader()
    for data in loader:
        print(len(data))  #  in_data, out_data, tmp_seq_info, idx
        input_data = build_input(dataset, utterance, data[0])
        assert torch.all(input_data == data[0])


if __name__ == "__main__":
    model = "joint_tts_nsf_hifigan"
    dataset = "debug_data"  # contains only one utterance
    utterance = "251-118436-0019"
    # debug_model(model, dataset)
    run_main(model)
