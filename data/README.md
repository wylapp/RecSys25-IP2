# Data Preparation
Due to the copyright issue and the large volume of the files, we cannot 
include datasets in our repository. We provide a simple utility to download 
the data bundles and arrange files into the correct directory structure.

To use the data acquiring utility, please run the following command:
```bash
source ./data_acquire.sh
```

Before using the MIND dataset, please visit the MIND dataset website and make sure 
you **read and consent** to the *Microsoft Research License Terms*.

## Folder Structure Example
Take MIND-small as an example:
```bash
data
├── MIND_small
    ├── test
    │   ├── behaviors.tsv
    │   ├── entity_embedding.vec
    │   ├── news.tsv
    │   └── relation_embedding.vec
    └── train
        ├── behaviors.tsv
        ├── entity_embedding.vec
        ├── news.tsv
        └── relation_embedding.vec
```
You can of course place the dataset into other directories.
Just change the `data_dir` in `config.yaml` to the directory where you want place the data.