# Martest-pipeline  

This repository is the implementation of [Testing Reinforcement Learning using NLP techniques](https://repositorio.uniandes.edu.co/entities/publication/80630919-0cd4-42bf-a52a-9b6803ac4209). It is based on how we can test reinforcement learning agents using a network (a conditioned transformer) usually applied to natural language processing problems.  

Basically, in this type of testing we take the agent’s logs—the paths the agent followed during its deployment—and use them to train the network (conditioned transformer). The goal is to find scenarios where the agent may fail. This type of testing is described as **Data Testing**, a kind of testing where you don’t know the system’s code or architecture, and only need data about its performance.  

The idea of the **Conditioned Transformer** is that the network learns the paths, rewards, and other objective functions that you define, and then—with the appropriate settings—predicts failure scenarios. You can see the architecture of the conditioned transformer here:  

![Conditioned Transformer](readme_docs/conditioned_transformer.png)  

As you can see, the conditioned transformer is composed of a network and GPT transformer components. These are implemented with the Hugging Face Transformers library and are trained end to end jointly.  

## MarTest: An NLP Approach for Test Sequence Generation  

Sometimes the state space of the agents is infinite and continuous, while the conditioned transformer (like almost all NLP models) works with **finite sequences**. Therefore, we need a way to bridge both worlds. We based this idea on a mathematical concept called *abstract classes*—or in this case, *abstract states*—and with this we can generate the input for the conditioned transformer. We deal with this in the first four modules:  

- 1_q_values_tables  
- 2_abstract_classe  
- 3_abstrct_episodes  
- 4_random_forest  

In the fifth module, we train the conditioned transformer. You can see this in the next figure:  

![Pipeline architecture](readme_docs/Pipeline_achitecture.png)  

For more information you can read more [here](https://repositorio.uniandes.edu.co/entities/publication/80630919-0cd4-42bf-a52a-9b6803ac4209).  
