# Learning to Generalize in Heterogeneous Federated Networks

In this repository, we implement **FedMC** and the baselines, including:

- **FedAvg** is the most classic method in federated learning, which averages all the model parameters from the selected clients at each communication round.
- **FedProx** provides a re-parametrization of FedAvg using a proximal term to regularize the local model parameters with the global model parameters in parameter space to address heterogeneity.
- **MOCHA**: in this method, each client is viewed as a task, and FL's objective is optimized in multi-task fashion, where each client is encouraged to be close to its neighboring clients, i.e., with similar data distribution. 
- **FedPer** proposes to learn a unified representation under the orchestration of global server, and the personalized layers are kept locally to capture clients' specific data distributions.
- **LG-FedAvg** jointly learns compact local features on each client and aggregates only the global classification model at each communication round, thus data heterogeneity can be explicitly modeled and reduced.
- **Per-FedAvg** is intended to find a global initial model and clients achieve personalized models through fine-tuning it on their private datasets, which is similar to MAML.
- **FedRep**: similar to FedPer, it learns both a unified representation and local personalized representations. Differently, in FedRep, each client optimizes base model and head model alternately, while FedPer optimizes all parameters simultaneously.
- **FedFomo** allows clients to federated only with their relevant clients. Before local optimization, each client initializes its local model using a linear combination of global models. 
## Datasets

We use four widely used federated benchmark datasets to simulate heterogeneous federated settings, including MINIST, CIFAR-10, CIFAR-100, and HAR.

| Dataset   | Task                 | #Clients | #Samples | #Samples per client | Classes | Base model |
|-----------|----------------------|----------|----------|---------------------|---------|------------|
| MNIST     | Digit classification | 100      | 70000    | 700                 | 10      | 2CNN + 2FC |
| CIFAR-10  | Image classification | 100      | 60000    | 600                 | 10      | 4CNN + 2FC |
| CIFAR-100 | Image classification | 100      | 60000    | 600                 | 100     | 4CNN + 2FC |
| HAR       | Activity recognition | 30       | 10269    | 342.3               | 6       | 4CNN + 2FC |

## Requirements
- Install the libraries listed in requirements.txt
```bash
pip install -r requirements.txt
```
## Experiments

We provide the bash scripts to run all experiments.

```bash
cd algorithms/
bash run.sh
```
