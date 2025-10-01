# Martest-pipeline

This repository is the implementation of the [Testing Reinforcment Learning using NLP techniques](https://repositorio.uniandes.edu.co/entities/publication/80630919-0cd4-42bf-a52a-9b6803ac4209) and this is based in how we can test reinforcement learning agents using a network (conditioned transformer) usually used in problems of natural language processing. 

Basically, in this type of testing we take the agent's logs of paths that took the agent in his own deployment and we take these for training of the network (conditioned transformer) with the goal of find scenarios where the agent may could fail. This type of testing is described as **Data Testing** a type of testing where you don't know the code or architecture of the system and only you need data of the system's performance. 

The idea of the **Conditioned transformer** is that the network learn the paths,rewards and other objective's functions that you define for then with the appropiate settings to predict failure scenarios. You can see the architecture of the conditioned transformer here: 

![Conditioned Transformer](readme_docs/conditioned_transformer.png)
