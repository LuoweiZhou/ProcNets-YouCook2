# Procedure Segmentation Networks (ProcNets)
This repo hosts the source code (Torch) for our work on procedure segmentation and YouCook2 dataset

* The large-scale cooking video dataset YouCook2 is available at [YouCook2 website](http://youcook2.eecs.umich.edu)
* Our AAAI18 oral paper is available at [arXiv preprint](https://arxiv.org/abs/1703.09788)

### Setup
1. Install [Lua Torch](http://torch.ch/docs/getting-started.html), which also contains packages such as nn, nngraph, cutorch etc.
2. Install [csvigo](https://github.com/clementfarabet/lua---csv) to read/write .csv files
3. Download the [YouCook2 dataset](http://youcook2.eecs.umich.edu/download)

### Feature
We provided ResNet-34 feature for [YouCook2](http://youcook2.eecs.umich.edu/download). If you want to extract feature from raw videos on your own, follow the instructions: i) Use or modify script/video2frame\_yc2.sh and script/videosample.py to sample frames, ii) Run extract\_cnnfeat\_resnet\_mscoco.lua which extracts feature for each frame.

### Training and validation
`train_bilstm_seq.lua` is the main file for training and validation. To load your data, make sure you specify the data path which is determined by `-image_folder`, `-train_data_folder`, `-val_data_folder` and `-ann_file`. You also need specify video info files at `-train_vidinfo_file` and `-val_vidinfo_file`. An example on model training:
```
th train_bilstm_seq.lua -id my_procnets -mp_scale_h 8 -mp_scale_w 5 -save_checkpoint_every 10000 -max_iters 120000 -learning_rate 4e-5
```
where the option `-save_checkpoint_every` determines the frequency for validation. The metrics used in validation include mIoU and Jacc, and the model with the highest Jacc will be stored under directory `-checkpoint_path`.

To load the pre-trained model for validation, run:
```
th train_bilstm_seq.lua -id eval-my_procnets -mp_scale_h 8 -mp_scale_w 5 -max_iters 1 -start_from /path/to/your/model
```
You can download our [pre-trained model(59MB)](http://youcook2.eecs.umich.edu/static/pre-trained-model/model_id_procnets-lr4e-5.t7). To evaluate the model in terms of precision and recall, run `script/eval_recall_precision.py`

Note: training is slow with the current implementation (2 days on NVIDIA GTX 1080Ti) and can be further optimized. Feel free to contribute to the repo!

### Visualization
We provide simple visualization of the generated segments, which can be enabled by setting `-vis true`. Run `script/plot_losses.py` to plot the training loss and validation accuracy.

### Others
Our code is mainly based on [Neuraltalk2](https://github.com/karpathy/neuraltalk2) and [Facebook ResNet](https://github.com/facebook/fb.resnet.torch) (thanks to both for releasing their code!). We might release a PyTorch version of ProcNets, please stay tuned! Our follow-up work on dense video captioning (e.g., recipe generation) can be found on my [website](http://luoweizhou.net/cv.html) soon.

Please [contact me](http://luoweizhou.net/contact.html) if you have any trouble running the code. Please cite the following paper if you are using the code.

```
@article{zhou2017procnets,
  title={Towards Automatic Learning of Procedures from Web Instructional Videos},
  author={Zhou, Luowei and Xu, Chenliang and Corso, Jason J},
  journal={arXiv preprint arXiv: 1703.09788}
}
```
