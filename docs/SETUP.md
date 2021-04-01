

Note: this repository is the Social-NCE implementation based on the *latest* [Social-STGCNN official code](https://github.com/abduallahmohamed/Social-STGCNN) as of submission date. 

### Environment Setup ###

Note: require python 3.6
```bash
conda create --name social-stgcnn python=3.6 -y
source activate social-stgcnn
pip install -r requirements.txt
```

### Data Setup ###

The dataset is from [Social-STGCNN official repository](https://github.com/abduallahmohamed/Social-STGCNN/tree/master/datasets) and stored at `/datasets/`. Please note that the dataset processing takes some time when running the code the first time, which is once and for all.

To guarantee the dataset completeness, you could either:

(1) clone the original repository, copy & paste the `datasets` folder to project root directory, 

or (2) click on the DownGit [link](https://downgit.github.io/#/home?url=https://github.com/abduallahmohamed/Social-STGCNN/tree/master/datasets) to get datasets and unzip to project root directory.