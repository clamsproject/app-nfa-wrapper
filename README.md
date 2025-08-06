# NFA Wrapper

## Description

Wraps the [NVIDIA NeMo Forced Aligner tool](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/nemo_forced_aligner.html) 
to temporally align transcribed text with its audio source. 

## User instructions

General user instructions for CLAMS apps are available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp).

### System requirements

The NFA tool requires a local installation of the [NeMo Framework](https://github.com/NVIDIA/NeMo#installation).
For best results, follow the GitHub directions for installing nemo_toolkit via the `git clone` method. 
Models are only required from the `nemo_toolkit['asr']` module.
The current version of the app is based on NFA version 1.20.0.

Before running the app, the `NEMO_PATH` environment variable **must** be set with the absolute path of 
the NeMo directory, e.g. with `export NEMO_PATH=/path/to/NeMo` if using Linux.

This app requires Python 3.10.12 or higher. For local installation of required Python modules, see [requirements.txt](requirements.txt).

### Configurable runtime parameters

For the full list of parameters, please refer to the app metadata from the [CLAMS App Directory](https://apps.clams.ai) or the [`metadata.py`](metadata.py) file in this repository.

### Input and output details

This app accepts an empty MMIF file with the file locations of the required [`AudioDocument`](https://mmif.clams.ai/vocabulary/AudioDocument/v1/)/[`VideoDocument`]('https://mmif.clams.ai/vocabulary/VideoDocument/v1/') 
and [`TextDocument`]('https://mmif.clams.ai/vocabulary/TextDocument/v1/') sources.

The app outputs a [`Token`](http://vocab.lappsgrid.org/Token.html) annotation corresponding to each 
whitespace-delimited token in the source transcript, as well as a [`TimeFrame`](https://mmif.clams.ai/vocabulary/TimeFrame/v6/) 
annotation identifying the audio segment where it appears and an [`Alignment`](https://mmif.clams.ai/vocabulary/Alignment/v1/) 
annotation linking the `Token` and `TimeFrame`.
For more details, see the `output` section of the app metadata.