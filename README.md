# Align Your Tangent

Official PyTorch implementation of [Align Your Tangent: Training Better Consistency Models via Manifold-Aligned Tangents](https://arxiv.org/abs/2510.00658) by [Beomsu Kim](https://scholar.google.co.kr/citations?user=TofIFUgAAAAJ&hl=en)\*, [Byunghee Cha](https://openreview.net/profile?id=~ByungHee_Cha1)\*, and [Jong Chul Ye](https://scholar.google.com/citations?user=HNMjoNEAAAAJ&hl=en) (\*Equal contribution).

Consistency Models (CMs), which are trained to be consistent on diffusion or probability flow ordinary differential equation (PF-ODE) trajectories, enable one or two-step flow or diffusion sampling.
However, CMs typically require prolonged training with large batch sizes to obtain competitive sample quality.

<p align="center">
  <img src="https://github.com/1202kbs/AYT/blob/main/assets/main_figure.png"  width="40%" height="40%" />
</p>

<p align="center">
  <img src="https://github.com/1202kbs/AYT/blob/main/assets/tangents.png"  width="90%" height="90%" />
</p>

In this paper, we examine the training dynamics of CMs near convergence and discover that CM tangents -- CM output update directions -- are quite oscillatory, in the sense that they move parallel to the data manifold, not towards the manifold.
To mitigate oscillatory tangents, we propose a new loss function, called the **manifold feature distance (MFD)**, which provides manifold-aligned tangents that point toward the data manifold.

<p align="center">
  <img src="https://github.com/1202kbs/AYT/blob/main/assets/main.png"  width="40%" height="40%"/>
</p>

Consequently, our method -- dubbed **Align Your Tangent (AYT)** -- can accelerate CM training by orders of magnitude and even out-perform the learned perceptual image patch similarity metric (LPIPS).
Furthermore, we find that our loss enables training with extremely small batch sizes without compromising sample quality. **For instance, on CIFAR10, we achieve 10 times faster convergence and competitive FIDs with 1/8 batch size (bs), as shown in the above figure.**

## Training Consistency Models

### How to Use Weights and Biases

If you want to use wandb to track your experiments,

1. Log into your wandb account. See this [link](https://docs.wandb.ai/quickstart/) for a quick how-to.
2. Go to ``src/ayt/utils.py`` and enter your project name in the ``init_wandb`` function.
3. Set ``exp.use_wandb=True`` in training bash scripts.

### Training on CIFAR10

1. To download required assets for training such as FID statistics and pretrained models, run
    ```
    bash bash_scripts/cifar10/prep.sh
    ```

2. Go to ``src/ayt/constants.py`` and write absolute paths to ``configs``, ``data``, and ``results`` directories in ``CONFIG_ROOT``, ``DATA_ROOT``, ``RESULT_ROOT`` variables.

3. To train your own classifier, run
    ```
    bash bash_scripts/cifar10/train_classifier.sh
    ```
    To run Easy Consistency Training (ECT), run
    ```
    bash bash_scripts/cifar10/train_ecm.sh
    ```
    To run ECT with Align Your Tangent (AYT), run
    ```
    bash bash_scripts/cifar10/train_ecm_ayt.sh
    ```
    
### Training on ImageNet64

Coming Soon!

## References

If you find this paper useful for your research, please consider citing

```bib
@article{
  kim2025ayt,
  title={Align Your Tangent: Training Better Consistency Models via Manifold-Aligned Tangents},
  author={Beomsu Kim and Byunghee Cha and Jong Chul Ye},
  journal={arXiv preprint arXiv:2510.00658},
  year={2025}
}
```

## Acknowledgements

Our source code is based on [EDM](http://github.com/NVlabs/edm), [ECT](https://github.com/locuslab/ect), and [pytorch-fid](https://github.com/mseitzer/pytorch-fid). Thank you!

## Contact

Feel free to contact us through mail :)

- Beomsu Kim: beomsu.kim@kaist.ac.kr
- Byunghee Cha: paulcha1025@kaist.ac.kr
