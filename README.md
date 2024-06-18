# Generative Models for Visual Signals

This project aims to take a deep look into the implementation of accelerating the initialization of Denoising Diffusion Probabilistic Model (DDPM) training process by applying the initial prior strategy of Deep Image Prior (DIP).

To know more about this project (assignment), you could read the git commit messages in this repository, and the `report.pdf` (it is written in English).

## Development Environment

* OS: Windows 10 Home
* GPU: NVIDEA GeForce RTX 3070
* Environment: Anaconda 2023.09
* Python: 3.11.5
* PyTorch: 2.2.2
* torchvision: 0.17.2
* scikit-image: 0.22.0

## How to run this project

The main structures of the training and evaluation code are in notebook `main.ipynb`. I have divided them into two sections with the Markdown titles.

If you want to change the number of epochs in DIP model, you can change the variable `num_iter` in the code with "DIP training structure setup" title. On the other hand, for the number of epochs in DDPM, you can change the variable `num_epochs` in the code blocks after importing the libraries with the title for the start of DDPM.

For availabling the DIP output noise as the target noise of loss function, which means making DIP as a part of the DDPM architecture, you can set `WITH_DIP_PRIOR` variable with 'True' (currently, I set it True, so if you don't want to make the DDPM training incoporate with DIP output, then I suggest you to set it False).
