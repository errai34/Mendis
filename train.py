import sys
import os
import json
import argparse
import socket
import itertools
from time import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributions.multivariate_normal import MultivariateNormal
torch.set_default_tensor_type('torch.cuda.FloatTensor')

import nf.spline_flows as spf
import nf.flows as nfls

import GPUtil
from GPUtil import showUtilization as gpu_usage
import gc

print("Initial GPU Usage")

def prep_data(config):
    #-----------------------------------------------------------------------------------------------------------------
    # The Data
    conditional = config["conditional"]
    spectra = np.load("./data/new_Xuncond_data.npy")
    spectra = spectra.T
    print(spectra.shape)

    spectra = torch.Tensor(spectra[:, :config["window"]])
    spectra = spectra - 0.5
    dim = spectra.shape[-1]

    print("spectra dim is", dim)
    print(spectra.shape)
    labels = np.load("./data/new_yuncond_data.npy")
    print("labels shape", labels.shape)

    # conditioning on teff, logg, feh
    y = np.array([labels[:, 0], labels[:, 1], labels[:, 18]]).T
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 3)
    print(y.shape)

    cond_dim = y.shape[-1]
    print("y dim is:", cond_dim)

    if not conditional:
        new_spectra = torch.cat((spectra, y), 1)
        y = None
        print(new_spectra.size())
    else:
        new_spectra = spectra

    dim = new_spectra.shape[-1]
    print("New dim of the input vector is", dim)

    gc.collect()
    return new_spectra, y


def prep_model(dim, context, config, gpu=None):

    if gpu is not None:
        torch.cuda.set_device(gpu)
    base_mu, base_cov = torch.zeros(dim), torch.eye(dim)
    prior = MultivariateNormal(base_mu, base_cov)
    model = optimizer = None

    # Configure the normalising flow

    nfs_flow = spf.NSF_CL
    nflows = config["nflows"]
    hidden_dim = config["hidden_dim"]
    K = config["K"]
    B = config["B"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]

    flows = [
        nfs_flow(dim=dim, K=K, B=B, hidden_dim=hidden_dim)
        for _ in range(nflows)
    ]  
    convs = [nfls.Invertible1x1Conv(dim=dim) for _ in flows]
    norms = [nfls.ActNorm(dim=dim) for _ in flows]
    flows = list(itertools.chain(*zip(norms, convs, flows)))

    # Initialise the model
    model = nfls.NormalizingFlowModel(prior, flows)

    if gpu is not None:
        model = model.cuda()

        model = DDP(model, device_ids=[gpu]) 

    optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  
    print("number of params: ", sum(p.numel() for p in model.parameters()))

    return model, optimiser

# Training run
# -------------------------------------------------------------------------------------
def run_model(model, optimiser, train_loader, config):
    t0 = time()

    model.train()
    print("Started training")
    loss_history=[]

    n_epochs = config["n_epochs"]
    conditional = config["conditional"]
    batch_count = 0
    for k in range(n_epochs):
        for batch_idx, data_batch in enumerate(train_loader):
            bt0 = time()
            x, y = data_batch if conditional else (data_batch, None)

            zs, prior_logprob, log_det = model(x, context=y)

            logprob = prior_logprob + log_det
            loss = -torch.mean(logprob)  

            model.zero_grad()

            loss.backward()
            optimiser.step()
            loss_history.append(loss.item())
            bt1 = time()

            ellapsed = bt1 - bt0
            print(f'Elapsed batch time: {ellapsed:.1f} s')
            batch_count += 1

        print('epoch done')


        if k % 20 == 0:
            print("Loss at step k =", str(k) + ":", loss.item())
            print("GPU usage after deleting some more of them pesky tensors at iteration step k after emptying cache", str(k))
            gpu_usage()

        # return model

    t1 = time()
    print(f'Elapsed time: {t1-t0:.1f} s')

    return model

def get_samples(model, config):

    with torch.no_grad():

        n_samples = config["n_samples"]

        conditional = config["conditional"]
        if conditional:
            conditions = config["conditions"]
            cont = np.ones((n_samples, len(conditions)))
            for i in range(len(conditions)):
                cont[:, i] = conditions[i]

            cont = torch.tensor(cont, dtype=torch.float32).reshape(-1, 3).cuda()
        else:
            cont = None

        zs = model.module.sample(n_samples, context=cont)
        #zs = model.sample(n_samples, context=cont)
        z = zs[-1]
        z = z.to('cpu')
        z = z.detach().numpy()
        np.save('./output/samples' + config['savepath'] + '_noParallel.npy', z)

        fig = plt.figure(figsize=(14, 4))

        for i in range(10):
            plt.plot(z[i, :2405])

        plt.savefig('./output/' + config['savepath'] + '_noParallel.png')

        # get_corr_matrix(z, 800)


def get_model_path(config):
    return './models/model_' + config['savepath'] + '.pth'


def train(gpu, args, config):

    world_rank = args.rank * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=world_rank)
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    maddr = os.environ['MASTER_ADDR']
    print(f'Hello from node {args.rank} gpu {gpu} ip {ip} with master {maddr}, world rank {world_rank}')


    print("GPU usage before loading the spectra into train loader...")
    gpu_usage()

    conditional = config["conditional"]
    spectra, y = prep_data(config)
    model, optimiser = prep_model(spectra.shape[-1], y, config, gpu)

    dataset = TensorDataset(spectra, y) if conditional else spectra

    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    train_sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=world_rank)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False, sampler=train_sampler)

    del dataset
    print("GPU Usage after deleting new spectra in train")
    gpu_usage()

    model = run_model(model, optimiser, train_loader, config)

    print("GPU Usage after emptying the cache")
    torch.cuda.empty_cache()
    gc.collect()
    gpu_usage()

    if world_rank == 0:
        #SAVE THE MODEL BY THE MASTER ONE

        torch.save(model.module, get_model_path(config))
        print('PATH is', get_model_path(config))
        get_samples(model, config)


def proclaim_master_self(args):
    # proclaim self as master in the absence of dominance
    if "MASTER_ADDR" not in os.environ:
        if args.nodes != 1 or args.rank != 0:
            raise Exception("I am unfit to rule the world. Sepuku.")

        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)

        os.environ["MASTER_ADDR"] = ip
        os.environ["MASTER_PORT"] = "8888"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int)
    parser.add_argument('-g', '--gpus', default=1, type=int)
    parser.add_argument('-r', '--rank', default=0, type=int)
    parser.add_argument('-c', '--config')   
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    proclaim_master_self(args)

    args.world_size = args.gpus * args.nodes
    mp.spawn(train, nprocs=args.gpus, args=(args,config,))
