modularity_vitality
==============================

Materials for "Measuring Node Contribution to Community Structure with Modularity Vitality" ([paper here](https://arxiv.org/abs/2003.00056v3))


Modularity Vitality itself is run as:
```
src.analysis.modularity_vitality.modularity_vitality(g, part)
```
where `g` is an igraph Graph, and part is an igraph partition. This function returns a list containing the vitalities for each of the nodes.

The results from the paper can be re-computed using the "Full Results" notebook.

To setup the enviornment to run these notebooks, first create a virtual enviornment using the requirements found in "requirements.txt", then run `pip install –e .` in the activated enviornment. 


