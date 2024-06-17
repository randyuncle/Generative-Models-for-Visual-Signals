import torch
import numpy as np

from tqdm import tqdm

from deep_image_prior.utils import *

def broadcast(values, broadcast_to):
    values = values.flatten()

    while len(values.shape) < len(broadcast_to.shape):
        values = values.unsqueeze(-1)

    return values

def _warmup_beta(beta_start, beta_end, num_timesteps, warmup_frac):
    betas = beta_end * torch.ones(num_timesteps)
    warmup_time = int(num_timesteps * warmup_frac)
    betas[ : warmup_time] = torch.linspace(beta_start, beta_end, warmup_time)
    return betas

def get_beta_schedule(schedular, beta_start, beta_end, num_timesteps):
    betas = None
    if schedular == 'quad':
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
    elif schedular == 'linear':
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
    elif schedular == 'warmup10':
        betas = _warmup_beta(beta_start, beta_end, num_timesteps, 0.1)
    elif schedular == 'warmup50':
        betas = _warmup_beta(beta_start, beta_end, num_timesteps, 0.5)
    elif schedular == 'const':
        betas = beta_end * torch.ones(num_timesteps)
    elif schedular == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / torch.linspace(num_timesteps, 1, num_timesteps)
    
    assert betas.shape == (num_timesteps, )
    return betas

# To be note that, the implementation of the DDPM is simplified compared to the original structure
class DDPM():
    def __init__(self, schedular='linear', beta_start=1e-4, beta_end=1e-2, num_timesteps=1000):
        # Betas settings refers to the implementation in the source code,
        # where the author set them with np.linspace, mostly.
        # In here, it only implement the case when `beta_schedule == 'linear'`
        self.betas = get_beta_schedule(schedular, beta_start, beta_end, num_timesteps)
        self.num_timesteps = num_timesteps
        
        self.alphas = 1 - self.betas
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)
        
        self.alphas_sprt = torch.sqrt(self.alphas_hat)
        self.alphas_one_minus_sqrt = torch.sqrt(1. - self.alphas_hat)

    def forward_diffusion(self, images, timesteps, dip_model, batch_size) -> tuple[torch.Tensor, torch.Tensor]:
        """The forward function of diffusion probabilistic model
        
        Args:
            images: The input data
            timesteps: Current forwarding time
        """
        noise = dip_model(get_noise(32, "noise", (32, 32)).to(images.device).detach()).repeat(batch_size, 1, 1, 1).to(images.device) \
                if dip_model else torch.randn(images.shape).to(images.device)
        self.alphas_hat = self.alphas_hat.to(images.device)
        alpha_hat = self.alphas_hat[timesteps]
        # alphas_sprt = self.alphas_sprt[timesteps].to(images.device)
        # alphas_one_minus_sqrt = self.alphas_one_minus_sqrt[timesteps].to(images.device)
        
        alpha_hat = broadcast(alpha_hat, images)

        # The calculation of the forward loss function `Lsimple(Θ)`, 
        # which is the equation (14) in the original paper
        return torch.sqrt(alpha_hat) * images + torch.sqrt(1 - alpha_hat) * noise, noise

    def reverse_diffusion(self, model, noisy_images, timesteps):
        """The reverse function of diffusion probabilistic model.
        Generates the predictions of the input images.

        Args:
            model: The input DDPM model
            noisy_images: The input noisy image
            timesteps: the time steps
        """
        predicted_noise = model(noisy_images, timesteps)
        return predicted_noise

    @torch.no_grad()
    def sampling(self, ddpm_model, dip_model, initial_noise, batch_size, device, save_all_steps=False):
        """
        Algorithm 2 from the paper https://arxiv.org/pdf/2006.11239.pdf
        Seems like we have two variations of sampling algorithm: iterative and with reparametrization trick (equation 15)
        Iterative assumes you have to denoise image step-by-step on T=1000 timestamps, while the second approach lets us
        calculate x_0 approximation constantly without gradually denosing x_T till x_0.
        """
        image = initial_noise
        images = []
        for timestep in tqdm(range(self.num_timesteps - 1, -1, -1)):
            # Broadcast timestep for batch_size
            ts = timestep * torch.ones(image.shape[0], dtype=torch.long, device=device)
            predicted_noise = self.reverse_diffusion(ddpm_model, image, ts)
            
            # The variable for the given timestep
            beta_t = self.betas[timestep].to(device)
            alpha_t = self.alphas[timestep].to(device)
            alpha_hat = self.alphas_hat[timestep].to(device)

            # Algorithm 2, step 4: calculate x_{t-1} with alphas and variance.
            # Since paper says we can use fixed variance (section 3.2, in the beginning),
            # we will calculate the one which assumes we have x0 deterministically set to one point.
            alpha_hat_prev = self.alphas_hat[timestep - 1].to(device)
            posterior_variance = (1 - alpha_hat_prev) / (1 - alpha_hat) * beta_t
            
            noise = dip_model(get_noise(32, "noise", (32, 32)).to(image.device).detach()).repeat(batch_size, 1, 1, 1).to(image.device) \
                    if dip_model else torch.randn(image.shape).to(image.device)

            variance = torch.sqrt(posterior_variance) * noise if timestep > 0 else 0

            image = torch.pow(alpha_t, -0.5) * (image -
                                                beta_t / torch.sqrt((1 - alpha_hat_prev)) *
                                                predicted_noise) + variance
            if save_all_steps:
                images.append(image.cpu())
        return images if save_all_steps else image