# [4/19/21] DEPRECATED -- please use [this repo](https://github.com/rpryzant/deconfounded-lexicon-induction) instead 

This repo is no longer under active development. 

# Deconfounded Lexicon Induction

This repo contains code for the paper, "Deconfounded Lexicon Induction for Interpretable Social Science".

You can read the paper [here](https://nlp.stanford.edu/pubs/pryzant2018lexicon.pdf). It outlines the algorithms implemented in this repo.

The project website is [here](https://nlp.stanford.edu/projects/deconfounded-lexicon-induction/).

## Installation

`pip install -r requirements.txt`

## Usage

`python main.py --config sample_config.json [--train] [--test]`

Logs, models, features, and summary files will be written in the config's `working_dir`. See `sample_config.json` for an example config and explanation. 

