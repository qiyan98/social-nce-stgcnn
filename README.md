## Social-STGCNN with Social-NCE

<p align="center">
  <img src="docs/illustration.png" width="300">
</p>

This is an official implementation of the Social-NCE applied to the Social-STGCNN forecasting model.

**[Social NCE: Contrastive Learning of Socially-aware Motion Representations](https://arxiv.org/abs/2012.11717)**
<br>by
<a href="https://sites.google.com/view/yuejiangliu/">Yuejiang Liu</a>,
<a href="https://qiyan98.github.io/">Qi Yan</a>,
<a href="https://people.epfl.ch/alexandre.alahi/?lang=en/">Alexandre Alahi</a> at
<a href="https://www.epfl.ch/labs/vita/">EPFL</a>
<br>

TL;DR: Contrastive Representation Learning + Negative Data Augmentations &#129138; Robust Neural Motion Models

> * Rank in 1st place on the [Trajnet++ challenge](https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge/leaderboards) since November 2020 to present
> * Significantly reduce the collision rate of SOTA [human trajectroy forecasting models](https://github.com/StanfordASL/Trajectron-plus-plus)
> * SOTA on imitation / reinforcement learning for [autonomous navigation in crowds](https://github.com/vita-epfl/CrowdNav)

Please check out our code for experiments on different models as follows:  
**[Social NCE + STGCNN](https://github.com/qiyan98/social-nce-stgcnn)  |  [Social NCE + CrowdNav](https://github.com/vita-epfl/social-nce-crowdnav)  |  [Social NCE + Trajectron](https://github.com/YuejiangLIU/social-nce-trajectron-plus-plus)**

### Preparation
Setup environments following the [SETUP.md](docs/SETUP.md)

### Training & Evaluation

**Train from scratch:**

```bash
bash train_snce.sh && python test.py --prefix snce # for social-nce results
bash train_random_sampling.sh && python test.py --prefix random-sampling # for social-nce + random sampling
```

See generated `results_default.csv` for detailed results.

**Test the pretrained model:**

```bash
python test.py --mode snce  # our social-nce pretrained model
python test.py --mode random-sampling # our social-nce method + random sampling, for ablation study
python test.py --mode baseline  # baseline pretrained model, obtained from official repo.
```

See generated `results_default.csv` for detailed results.

### Basic Results

The script above results in the following results (on NVIDIA GeForce RTX 3090). The result may subject to mild variance on different GPU devices. More details will be released soon!

<table>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<thead>
  <tr>
    <th rowspan="2">Scene</th>
    <th colspan="3">Social-STGCNN w/o Ours</th>
    <th colspan="3">Social-STGCNN w/ Ours</th>
  </tr>
  <tr>
    <td align="center">ADE</td>
    <td align="center">FDE</td>
    <td align="center">COL</td>
    <td align="center">ADE</td>
    <td align="center">FDE</td>
    <td align="center">COL</td>
  </tr>
</thead>
<!-- TABLE BODY -->
<tbody>
  <tr>
    <td align="center">ETH</td>
    <td align="center">0.732</td>
    <td align="center">1.223</td>
    <td align="center">1.33</td>
    <td align="center">0.664</td>
    <td align="center">1.224</td>
    <td align="center">0.61</td>
  </tr>
  <tr>
    <td align="center">HOTEL</td>
    <td align="center">0.414</td>
    <td align="center">0.687</td>
    <td align="center">3.82</td>
    <td align="center">0.435</td>
    <td align="center">0.678</td>
    <td align="center">3.35</td>
  </tr>
  <tr>
    <td align="center">UNIV</td>
    <td align="center">0.489</td>
    <td align="center">0.912</td>
    <td align="center">9.11</td>
    <td align="center">0.473</td>
    <td align="center">0.879</td>
    <td align="center">6.44</td>
  </tr>
  <tr>
    <td align="center">ZARA1</td>
    <td align="center">0.333</td>
    <td align="center">0.525</td>
    <td align="center">2.27</td>
    <td align="center">0.325</td>
    <td align="center">0.515</td>
    <td align="center">1.02</td>
  </tr>
  <tr>
    <td align="center">ZARA2</td>
    <td align="center">0.303</td>
    <td align="center">0.480</td>
    <td align="center">6.86</td>
    <td align="center">0.289</td>
    <td align="center">0.482</td>
    <td align="center">3.37</td>
  </tr>
  <tr>
    <td align="center"><b>Average</b></td>
    <td align="center">0.454</td>
    <td align="center">0.765</td>
    <td align="center">4.70</td>
    <td align="center">0.437</td>
    <td align="center">0.756</td>
    <td align="center">2.96</td>
  </tr>  
</tbody>
</table>

### Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{liu2020snce,
  title   = {Social NCE: Contrastive Learning of Socially-aware Motion Representations},
  author  = {Yuejiang Liu and Qi Yan and Alexandre Alahi},
  journal = {arXiv preprint arXiv:2012.11717},
  year    = {2020}
}
```

### Acknowledgement

Our code is developed upon the official implementation of [Social-STGCNN](https://github.com/abduallahmohamed/Social-STGCNN).