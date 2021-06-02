# 📈 BregmanLearning
Implementation of the inverse scale space training algorithms for sparse neural networks, proposed in **A Bregman Learning Framework for Sparse Neural Networks** [[1]](#1).
Feel free to use it and please refer to our paper when doing so.
```
@misc{bungert2021bregman,
      title={A Bregman Learning Framework for Sparse Neural Networks}, 
      author={Leon Bungert and Tim Roith and Daniel Tenbrinck and Martin Burger},
      year={2021},
      eprint={2105.04319},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## 💡 Method Description
Our Bregman learning framework aims at training sparse neural networks in an inverse scale space manner, starting with very few parameters and gradually adding only relevant parameters during training. We train a neural network <img src="https://latex.codecogs.com/svg.latex?f_\theta:\mathcal{X}\rightarrow\mathcal{Y}" title="net"/> parametrized by weights <img src="https://latex.codecogs.com/svg.latex?\theta" title="weights"/> using the simple baseline algorithm
<p align="center">
      <img src="https://latex.codecogs.com/svg.latex?\begin{cases}v\gets\,v-\tau\hat{\nabla}\mathcal{L}(\theta),\\\theta\gets\mathrm{prox}_{\delta\,J}(\delta\,v),\end{cases}" title="Update" />
</p>

where 
* <img src="https://latex.codecogs.com/svg.latex?\mathcal{L}" title="loss"/> denotes a loss function with stochastic gradient <img src="https://latex.codecogs.com/svg.latex?\hat{\nabla}\mathcal{L}" title="stochgrad"/>,
* <img src="https://latex.codecogs.com/svg.latex?J" title="J"/> is a sparsity-enforcing functional, e.g., the <img src="https://latex.codecogs.com/svg.latex?\ell_1" title="ell1"/>-norm,
* <img src="https://latex.codecogs.com/svg.latex?\mathrm{prox}_{\delta\,J}" title="prox"/> is the proximal operator of <img src="https://latex.codecogs.com/svg.latex?J" title="J"/>.

Our algorithm is based on linearized Bregman iterations [[2]](#2) and is a simple extension of stochastic gradient descent which is recovered choosing <img src="https://latex.codecogs.com/svg.latex?J=0" title="Jzero"/>. We also provide accelerations of our baseline algorithm using momentum and Adam [[3]](#3). 

The variable <img src="https://latex.codecogs.com/svg.latex?v" title="v"/> is a subgradient of <img src="https://latex.codecogs.com/svg.latex?\theta" title="weights"/> with respect to the *elastic net* functional 

<p align="center">
      <img src="https://latex.codecogs.com/svg.latex?J_\delta(\theta)=J(\theta)+\frac1\delta\|\theta\|^2" title="el-net"/>
</p>

and stores the information which parameters are non-zero.

## 🎲 Initialization

We use a *sparse initialization strategy* by initializing parameters non-zero with a small probability.
Their variance is chosen to avoid vanishing or exploding gradients, generalizing Kaiming-He or Xavier initialization.

## 🔬 Experiments

### Classification
<img src="https://user-images.githubusercontent.com/44805883/120520997-bdd0e880-c3d4-11eb-9743-166b097fe70b.png" width="700">


### NAS
<img src="https://user-images.githubusercontent.com/44805883/120520730-70547b80-c3d4-11eb-94f8-df36e24ad778.png" width="700">


## 📝 References
<a id="1">[1]</a> Leon Bungert, Tim Roith, Daniel Tenbrinck, Martin Burger. "A Bregman Learning Framework for Sparse Neural Networks." arXiv preprint arXiv:2105.04319 (2021). https://arxiv.org/abs/2105.04319

<a id="2">[2]</a> Woatao Yin, Stanley Osher, Donald Goldfarb, Jerome Darbon. "Bregman iterative algorithms for \ell_1-minimization with applications to compressed sensing." SIAM Journal on Imaging sciences 1.1 (2008): 143-168.

<a id="3">[3]</a> Diederik Kingma, Jimmy Lei Ba. "Adam: A Method for Stochastic Optimization." arXiv preprint arXiv:1412.6980 (2014). https://arxiv.org/abs/1412.6980
