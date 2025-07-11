# IP2
This is the official pytorch implementation of RecSys'25 paper 
"IP2: Entity-Guided Interest Probing for Personalized News Recommendation". 

## Requirements
If you use `pip`, please use `requirements.txt` to install the required packages. 
If you use `conda`, please use `ip2_env.yaml` to create the environment. 
Please make sure your GPU driver is correctly installed.

Our default environment is Debian 12 with pytorch 2.0.1+cu117.

## Prepare the Data
Due to the copyright issue, we cannot include data in our repository. 
Please refer to the README.md file in the `data` folder for more details 
on data preparation.

## Run the Code
Please carefully check the `config.yaml` file to make sure key parameters, 
data paths, and model parameters are correctly set. Once everything is ready, 
you can run the code with:
```bash
python main.py
```
In the first run, it will take a little bit longer time to process the data. 
You can use tensorboard to monitor the training process and the testing results.

## Cite our work!
If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{wu2025IP2,
	title={IP2: Entity-Guided Interest Probing for Personalized News Recommendation},
	author={Wu, Youlin and Sun, Yuanyuan and Zhang, Xiaokun and Xu, Bo and Yang, Liang and Lin, Hongfei},
	booktitle={Proceedings of the 19th ACM conference on recommender systems},
	year={2025},
    doi={10.1145/3705328.3748091}
}
```
