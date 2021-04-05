# Ultra-Fast-Lane-Detection
PyTorch implementation of the paper "[Ultra Fast Structure-aware Deep Lane Detection](https://arxiv.org/abs/2004.11757)".

Updates: Our paper has been accepted by ECCV2020.

# Install
Please see [INSTALL.md](./INSTALL.md)

# Get started
First of all, please modify `data_root` and `log_path` in your `configs/culane.py` or `configs/tusimple.py` config according to your environment. 
- `data_root` is the path of your CULane dataset or Tusimple dataset. 
- `log_path` is where tensorboard logs, trained models and code backup are stored. ***It should be placed outside of this project.***



***

For single gpu training, run
```Shell
python train.py configs/path_to_your_config
```
For multi-gpu training, run
```Shell
sh launch_training.sh
```
or
```Shell
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py configs/path_to_your_config
```
If there is no pretrained torchvision model, multi-gpu training may result in multiple downloading. You can first download the corresponding models manually, and then restart the multi-gpu training.

Since our code has auto backup function which will copy all codes to the `log_path` according to the gitignore, additional temp file might also be copied if it is not filtered by gitignore, which may block the execution if the temp files are large. So you should keep the working directory clean.
***

Besides config style settings, we also support command line style one. You can override a setting like
```Shell
python train.py configs/path_to_your_config --batch_size 8
```
The ```batch_size``` will be set to 8 during training.

***

To visualize the log with tensorboard, run

```Shell
tensorboard --logdir log_path --bind_all
```

# Demo on Custom Video
install required library
```Shell
pip install -r requirements.txt
```

```Shell
python test_custom.py configs/culane.py --test_model path_to_culane_model --video_path path_to_video
```
[link to culane_model](https://drive.google.com/u/0/uc?id=1lRwqeRjBhXSkkNb0kjfokRZZg82b3jC8&export=download)


# Citation

```BibTeX
@InProceedings{qin2020ultra,
author = {Qin, Zequn and Wang, Huanyu and Li, Xi},
title = {Ultra Fast Structure-aware Deep Lane Detection},
booktitle = {The European Conference on Computer Vision (ECCV)},
year = {2020}
}
```

# Thanks
Thanks zchrissirhcz for the contribution to the compile tool of CULane, KopiSoftware for contributing to the speed test, and ustclbh for testing on the Windows platform.
