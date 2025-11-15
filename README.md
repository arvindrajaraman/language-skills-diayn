# Discovering Skills with Language

**Arvind Rajaraman, Vivek Myers, Anca Dragan**

**Note**, this is an unpublished/unpolished repo. See diayn.py and agents/diayn_agent.py for code samples.

Paper: https://www.arvindrajaraman.com/data/dlsd.pdf

DLSD (Diverse Language-based Skill Discovery) is an unsupervised RL method that uses language embeddings as a semantic prior to guide skill discovery. We see that in long-horizon tasks, DLSD is able to compose elementary skills into complex ones better than traditional methods.

<p align="center">
  <img width="1115" height="207" alt="image" src="https://github.com/user-attachments/assets/53204c15-70ea-433e-80fa-f7060be06aac" />
</p>

Specifically, we tested our method on Crafter, a 2-D version of Minecraft with different achievements (like collecting materials, defeating enemies, and eating food). The image above shows the RL agent's performance over time on several tasks, with our method (Green) outperforming the oracle (Salmon) trained directly on the ground-truth reward! This is because the ground-truth reward is sparse and unstructured, and our language-based skill reward provides additional structure for learning. Other lines represent other unsupervised RL approaches.

<p align="center">
  <img width="880" height="425" alt="image" src="https://github.com/user-attachments/assets/235e5d02-43da-4a03-b207-8288abea60a9" />
</p>

The diagram above is a high-level overview of our learning system: (1) a discriminator network that guides discovery towards new skills, (2) a language embedding space that structures learning, and (3) a skill-conditioned policy that powers the agent.
