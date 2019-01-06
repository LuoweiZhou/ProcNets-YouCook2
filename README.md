# Procedure Segmentation Networks (ProcNets)
This repo hosts the source code (Torch) for our work on procedure segmentation and YouCook2 dataset

* The large-scale cooking video dataset YouCook2 is available at [YouCook2 website](http://youcook2.eecs.umich.edu)
* Our AAAI18 oral paper is available [here](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17344)


### Setup
1. Install [Lua Torch](http://torch.ch/docs/getting-started.html), which also contains packages such as nn, nngraph, cutorch etc.
2. Install [csvigo](https://github.com/clementfarabet/lua---csv) to read/write .csv files
3. Download the [YouCook2 dataset](http://youcook2.eecs.umich.edu/download)


### Feature
We provide ResNet-34 feature for 500 uniformly sampled RGB frames per video (see dataset [README](http://youcook2.eecs.umich.edu/static/YouCookII/youcookii_readme.pdf)). To extract feature on your own, follow the instructions: i) Adapt `script/video2frame\_yc2.sh` and `script/videosample.py` to sample frames, ii) Run `extract\_cnnfeat\_resnet\_mscoco.lua` to extract feature for each frame.


### Training and validation
`train_bilstm_seq.lua` is the main file for training and validation. To load your data, specify the data paths `-image_folder`, `-train_data_folder`, `-val_data_folder` and `-ann_file`. You also need specify video info files at `-train_vidinfo_file` and `-val_vidinfo_file`. An example on model training:
```
th train_bilstm_seq.lua -id my_procnets -mp_scale_h 8 -mp_scale_w 5 -save_checkpoint_every 10000 -max_iters 120000 -learning_rate 4e-5
```
where the option `-save_checkpoint_every` determines the frequency for validation. The metrics used in validation include mIoU and Jacc, and the model with the highest Jacc will be stored under directory `-checkpoint_path`.

Note: training is slow with the current implementation (2 days on NVIDIA GTX 1080Ti) and can be further optimized. We actively welcome pull requests.


### Testing
The model testing is integrated in the same script, so simply run:
```
th train_bilstm_seq.lua -id eval-my_procnets -mp_scale_h 8 -mp_scale_w 5 -max_iters 1 -start_from /path/to/your/model
```
Make sure you specify `-val_data_folder` and `-val_info_file` to the feature and duration info corresponding to the testing split.

We provide our [pre-trained model (59MB)](http://youcook2.eecs.umich.edu/static/pre-trained-model/model_id_procnets-lr4e-5.t7). The Jacc and mIoU scores are shown below. To evaluate the model in terms of precision and recall, refer to `script/eval_recall_precision.py`.

<table>
  <tr>
    <th></th>
    <th colspan="2">validation</th>
    <th colspan="2">test</th>
  </tr>
  <tr>
    <td>Method</td>
    <td>Jaccard</td>
    <td>mIoU</td>
    <td>Jaccard</td>
    <td>mIoU</td>
  </tr>
  <tr>
    <td>ProcNets-LSTM</td>
    <td>55.3</td>
    <td>40.9</td>
    <td>51.5</td>
    <td>38.0</td>
  </tr>
</table>


### Visualization
We provide simple visualization of the generated segments, which can be triggered by setting `-vis` to `true`. Run `script/plot_losses.py` to plot the training loss and validation accuracy.


### Acknowledgement
Our code is mainly based on [Neuraltalk2](https://github.com/karpathy/neuraltalk2) and [Facebook ResNet](https://github.com/facebook/fb.resnet.torch) (thanks to both for releasing their code!). We are releasing a PyTorch version of ProcNets soon, please stay tuned!

Please contact <luozhou@umich.edu> if you have any trouble running the code. Please cite the following paper if you are using the code.

```
  @inproceedings{ZhXuCoCVPR18,
    author={Zhou, Luowei and Xu, Chenliang and Corso, Jason J},
    title = {Towards Automatic Learning of Procedures From Web Instructional Videos},
    booktitle = {AAAI Conference on Artificial Intelligence},
    year = {2018},
    url = {https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17344}
  }
```
