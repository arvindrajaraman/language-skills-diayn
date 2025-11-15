# Discovering Skills with Language

**Arvind Rajaraman, Vivek Myers, Anca Dragan**

**Note**, this is an unpublished/unpolished repo. See diayn.py and agents/diayn_agent.py for code samples.

Unsupervised reinforcement learning (RL) algorithms promise to learn meaningful behaviors in the absence of a known reward function. In practice, these approaches fail to scale effectively to high-dimensional settings. The fundamental issue is that in the absence of additional structure, the unsupervised RL problem is underspecified: the best we can do is try and discover skills that cover the full space of behaviors, most of which will be meaningless. In many environments where we might wish to do skill discovery, language can provide precisely this additional structure, telling us which behaviors are meaningful to discover. We argue that applying the structure of pre-trained language embeddings as a prior for skill discovery enables us to learn more meaningful behaviors, demonstrating our approach (DLSD: Diverse Language-based Skill Discovery) in the open-world Crafter environment.
