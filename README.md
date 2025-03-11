# Learning to Localize Leakage of Cryptographic Sensitive Variables
By Jimmy Gammell, Anand Raghunathan, Abolfazl Hashemi, and Kaushik Roy

IN PROGRESS: writing README and updating code so that it is usable by other people.

Our paper can be found [here](https://arxiv.org/abs/2503.07464).

## Installation

We used Python 3.9.19. It seems like things break if the Python version is too recent.
1) Create a conda environment with `<conda/mamba/micromamba> create --name=leakage-localization python=3.9`.
2) Activate it with `<conda/mamba/micromamba> activate leakage-localization`.
3) Navigate to the project directory and install required packages with `pip3 install -r requirements.txt`.

## Usage

All experiments are run via argparse commands to `src/run_trial.py`. Additionally, we have a self-contained minimal working example in `TODO`.

## Citation

Please cite the following paper if you use our work:
```
@misc{gammell2025learninglocalizeleakagecryptographic,
      title={Learning to Localize Leakage of Cryptographic Sensitive Variables}, 
      author={Jimmy Gammell and Anand Raghunathan and Abolfazl Hashemi and Kaushik Roy},
      year={2025},
      eprint={2503.07464},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.07464}, 
}
```