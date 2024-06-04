## Dataset:
All data used in this paper are publicly available and can be accessed here:
* PDBbind v2016 and v2019: http://www.pdbbind.org.cn/download.php
* CASF2016 and CASF2013: http://www.pdbbind.org.cn/casf.php

## Requirments:
python==3.8.13
numpy==1.23.1
torch==1.11.0
torch_g2.eometric==2.0.4
rdkit==2022.03.5

## Process:
First, you can run "./protein_ligand_affinity/conversation.py" to generate the pocket.
Then, you can run "./protein_ligand_affinity/construct_graph.py" to create the graphs needed to run the models.
