import sys
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

import train as tsc
import torch
torch.set_default_tensor_type('torch.FloatTensor')

import multiprocessing as mp


def get_samples(idx, model, config, processes):
    with torch.no_grad():
        n_samples = config["n_samples"] // processes

        conditional = config["conditional"]
        if conditional:
            conditions = config["conditions"]
            cont = np.ones((n_samples, len(conditions)))
            for i in range(len(conditions)):
                cont[:, i] = conditions[i]

            cont = torch.tensor(cont, dtype=torch.float32).reshape(-1, len(conditions))#.cuda()
        else:
            cont = None

        print(idx, 'sampling')
        zs = model.sample(n_samples, context=cont)
        z = zs[-1]#.to('cpu')
        z = z.detach().numpy()
        print(idx, 'done')
        return z

def save_samples(config, z):
    np.save('./output/samples_parallel' + config['savepath'] + '.npy', z)
    print('saved output npy')

    fig = plt.figure(figsize=(14, 4))

    for i in range(10):
        plt.plot(z[i])

    plt.savefig('./output/parallel' + config['savepath'] + '.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    parser.add_argument('-p', '--processes')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # spectra, y = tsc.prep_data(config)
    # model, optimiser = tsc.prep_model(spectra.shape[-1], y, config)
    # load_model(model, config)
    model = torch.load(tsc.get_model_path(config), map_location=torch.device('cpu'))
    processes = int(args.processes)
    pool = mp.Pool(processes)
    
    procs = []
    for i in range(processes):
    	procs.append(pool.apply_async(get_samples, (i, model, config, processes)))
    
    ret = []
    for p in procs:
    	p.wait()
    	ret.append(p.get())
    ret = np.concatenate(ret)

    save_samples(config, ret)


