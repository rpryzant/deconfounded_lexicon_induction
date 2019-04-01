# Text Attribution

Code for [Interpretable Neural Architectures for Attributing an Ad’s Performance to its Writing Style](https://nlp.stanford.edu/pubs/pryzant2018emnlp.pdf)


## Setup

```
virtualenv ~/venv
source ~/venv/bin/activate
pip install -r requirements.txt
```

## Usage

```
python text_attribution.py --yaml_config sample_config.yaml
```

This command will run an experiment from start to finish: it will train all the
models you ask it to train, extract outputs, and analyze the outputs.

See `sample_config.yaml` for instructions on how to configure an experiment.

## Outputs

If you ran `sample_config.yaml`, your outputs will be in `~/Desktop/test_run`;
please take a look at what's in there:

*   `config.yaml`, a copy of the config for this run.
*   `np_data.npy`, preprocessed data that can be restored.
*   `summary.csv`, a list of models and results.
*   Directories for each model. Inside each directory is a `.txt` file
    containing the top-scoring words as decided by the corresponding model.

## Data Preparation

All data are expected to be pre-tokenized and formatted as tsv's.

Configure the system to your data in a `config.yaml` file. See
`sample_config.yaml` for instructions and an example.
