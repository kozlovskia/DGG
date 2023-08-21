import pickle as pkl

import numpy as np
import matplotlib.pyplot as plt
import torch

from common_utils import reparametrize


def main(args):
    vae = torch.load(args.read_dir / 'vae.pkl')
    gmm = torch.load(args.read_dir / 'GMM.pkl')

    vae.to(args.device)
    gmm.to(args.device)

    vae.eval()
    gmm.eval()

    fig_dim = 32 if args.dataset == 'cifar10' else 28
    fig, axs = plt.subplots(10, 10, figsize=(fig_dim, fig_dim))

    for i in range(args.n_clusters):
        mean = gmm.mean.data[0, :, i]
        logvar = (gmm.std.data[0, :, i] ** 2).log()
        gen_samples = [reparametrize(mean, logvar) for _ in range(10)]
        gen_samples = torch.cat(gen_samples, dim=0).reshape(-1, args.latent_dim)
        gen_samples = vae.get_recon(gen_samples).view(-1, *args.output_shape)
        # gen_samples = torch.sigmoid(gen_samples).data.cpu().numpy()
        gen_samples = gen_samples.data.cpu().numpy()

        for j in range(10):
            if args.dataset != 'cifar10':
                axs[i, j].imshow(gen_samples[j][0], cmap='gray')
            else:
                axs[i, j].imshow(np.transpose(gen_samples[j], (1, 2, 0)))
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(args.save_dir / 'generated_samples.png')
