# HAVA

Code for "HAVA: Hybrid Approach to Value-Alignment through Reward Weighing for Reinforcement Learning".

## Getting Started

Clone the repository:

```
git clone https://github.com/kvarys/HAVA.git
```

or

```
git clone git@github.com:kvarys/HAVA.git
```

Enter the project:

```
cd HAVA
mkdir models
mkdir results
mkdir results/logs
```

Set up the Python environment:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Train the HAVA policy with

```
python main.py hava 0.1 1.0 mix
```

or the safe/legal policy (no social norms) with

```
python main.py hava 0.1 1.0 safe
```

or the social policy (no safety / legal norms) with

```
python main.py hava 0.1 1.0 dd
```

Where 0.1 is alpha (essentially the number of steps before the reputation recovers), 1.0 is tau (tolerance) and "mix" means both safety/legal and social norms (HAVA), "safe" means just safety/legal norms but no social norms and "dd" means data-driven social norms only but not safety/legal norms.

Link to the paper:

[ifaamas](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2096.pdf)
[Arxiv](https://arxiv.org/abs/2505.15011)
[ACM](https://dl.acm.org/doi/10.5555/3709347.3743848)

Cite as:

```
@inproceedings{10.5555/3709347.3743848,
author = {Varys, Kryspin and Cerutti, Federico and Sobey, Adam and Norman, Timothy J.},
title = {HAVA: Hybrid Approach to Value-Alignment through Reward Weighing for Reinforcement Learning},
year = {2025},
isbn = {9798400714269},
publisher = {International Foundation for Autonomous Agents and Multiagent Systems},
address = {Richland, SC},
abstract = {Our society is governed by a set of norms which together bring about the values we cherish such as safety, fairness or trustworthiness. The goal of value alignment is to create agents that not only do their tasks but through their behaviours also promote these values. Many of the norms are written as laws or rules (legal / safety norms) but even more remain unwritten (social norms). Furthermore, the techniques used to represent these norms also differ. Safety / legal norms are often represented explicitly, for example, in some logical language while social norms are typically learned and remain hidden in the parameter space of a neural network. There is a lack of approaches in the literature that could combine these various norm representations into a single algorithm. We propose a novel method that integrates these norms into the reinforcement learning process. Our method monitors the agent's compliance with the given norms and summarizes it in a quantity we call the agent's reputation. This quantity is used to weigh the received rewards to motivate the agent to become value aligned. We carry out a series of experiments including a continuous state space traffic problem to demonstrate the importance of the written and unwritten norms and show how our method can find the value aligned policies. Furthermore, we carry out ablations to demonstrate why it is better to combine these two groups of norms rather than using either separately.},
booktitle = {Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems},
pages = {2096–2104},
numpages = {9},
keywords = {reinforcement learning, reward shaping, value alignment},
location = {Detroit, MI, USA},
series = {AAMAS '25}
}
```
