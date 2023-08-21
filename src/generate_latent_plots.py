import pickle as pkl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples

from DGG.src.DGG import to_numpy


def main(args):
    vae = torch.load(args.read_dir / 'vae.pkl')
    classifier = torch.load(args.read_dir / 'classifier.pkl')
    gmm = torch.load(args.read_dir / 'GMM.pkl')

    vae.to(args.device)
    classifier.to(args.device)
    gmm.to(args.device)

    vae.eval()
    classifier.eval()
    gmm.eval()

    image_train = torch.from_numpy(torch.load((args.save_dir / 'data') / '0_all_siamese.pkl')).unsqueeze(2)
    image_train = to_numpy(torch.cat([image_train],dim=-1))
    label_train = torch.load((args.save_dir / 'data') / 'label_all_siamese.pkl')

    print(image_train.shape, label_train.shape)

    x_mean, _ = vae.get_latent(torch.from_numpy(image_train[:,:,0]).to(args.device))
    print(x_mean.shape)
    label_pred = torch.max(gmm.compute_prob(x_mean), dim=-1)[-1].cpu().numpy()
    x_mean = to_numpy(x_mean.cpu())

    zs = x_mean
    ys = label_pred
    ts = label_train

    print('compute silhouette scores')
    silh_samples = silhouette_samples(zs, ys)
    print('done')

    silh_df = pd.DataFrame({'silh_samples': silh_samples, 'y': ys})
    silh_df.y = silh_df.y.astype(int).astype(str)
    silh_df.to_csv(args.save_dir / 'silh_samples.csv', index=False)

    fig = plt.figure(figsize=(20, 10))
    ax = plt.gca()
    sns.violinplot(y='silh_samples', x='y', data=silh_df, palette=sns.color_palette("Set3", 10), order=[str(i) for i in range(args.n_clusters)], ax=ax)
    ax.set(title=f'Silhouette samples for DGG latent space', ylabel='silhouette score', xlabel='predicted label')
    plt.tight_layout()
    plt.savefig(args.save_dir / 'silh_samples.png')

    print('fit tsne')
    tsne = TSNE(n_components=2, perplexity=args.perplexity)
    zs_tsne = tsne.fit_transform(zs)
    print('done', zs_tsne.shape)

    if args.latent_dim == 2:
        zs_tsne = zs

    plot_df = pd.DataFrame({'z_1': zs_tsne[:, 0], 'z_2': zs_tsne[:, 1], 'y': ys, 't': ts})
    plot_df[['y', 't']].astype(int).astype(str)
    plot_df.to_csv(args.save_dir / 'plot_df.csv', index=False)
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    fig.suptitle(f'TSNE representation of DGG latent space')
    sns.scatterplot(x='z_1', y='z_2', hue='y', data=plot_df, ax=axs[0], alpha=args.alpha, palette=sns.color_palette("Set3", args.n_clusters), legend=False)
    axs[0].set(title='predicted labels', xlabel='z_1', ylabel='z_2', xticklabels=[], yticklabels=[])
    sns.scatterplot(x='z_1', y='z_2', hue='t', data=plot_df, ax=axs[1], alpha=args.alpha, palette=sns.color_palette("Set3", args.n_clusters), legend=False)
    axs[1].set(title='true labels', xlabel='z_1', ylabel='z_2', xticklabels=[], yticklabels=[])
    plt.tight_layout()
    plt.savefig(args.save_dir / 'tsne.png')
