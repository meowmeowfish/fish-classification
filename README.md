# fish-classification

## Set up environment

### Install python 3

python version: 3.4.3

我使用[pyenv](https://github.com/pyenv/pyenv)來建立python虛擬環境

[pyenv教學](https://aji.tw/pyenv-python%E8%99%9B%E6%93%AC%E7%92%B0%E5%A2%83%E7%B5%95%E4%BD%B3%E5%88%A9%E5%99%A8/)

```
pyenv install 3.4.3
pyenv local 3.4.3
```

### Install necessary python packages with pip

```
pip install keras
pip install tensorflow
pip install h5py
pip install opencv-python
pip install imutils
pip install matplotlib
pip insatll sklearn
```

## Train the model:

```
python train_network.py --dataset images --model fuck-fish.model
```

## Test the model:

```
python test_network.py --model fuck-fish.model --image examples/pomfret1.jpg
python test_network.py --model fuck-fish.model --image examples/yellow3.jpg
```
