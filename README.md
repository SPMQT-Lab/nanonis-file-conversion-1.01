# NANONIS File Conversion

## What It Does

This repository is dedicated towards NANONIS raw data file conversion, to enable the usage of image analysis software that is built for one specific raw data file format, for several file formats. Currently the file conversions supported are:

* NANONIS images as .dat -> .sxm 
* NANONIS imagesd as .dat -> .png

We have two command line tools that implement the above. 

- `dat-png` converts image `.dat` files into PNG previews
- `dat-sxm` converts image `.dat` files into `.sxm` files

Both commands accept either a single `.dat` file or a directory of `.dat` files.

This public repository is derived from a refined lab-internal project hence the few commits. 

Co-authorship note: the CLI-oriented workflow was co-authored with Gustavo Campi.

## Installation

Clone the repository, enter it, and install it in editable mode:

```bash
git clone https://github.com/rohanplatts/nanonis-file-conversion.git
cd nanonis-file-conversion
python -m pip install -e .
```

That installs these commands into your active Python environment:

- `dat-png`
- `dat-sxm`


## Usage

### Use the built-in default paths

The repository ships with two sample `.dat` files in [data/sample_input](data/sample_input). With the defaults unchanged, you can run:

```bash
dat-png
dat-sxm
```

Default paths are defined here:

- [nanonis_tools/dats_to_pngs.py](nanonis_tools/dats_to_pngs.py)
  `DEFAULT_INPUT_DIR`, `DEFAULT_OUTPUT_DIR`
- [nanonis_tools/dat_sxm_cli.py](nanonis_tools/dat_sxm_cli.py)
  `DEFAULT_INPUT_DIR`, `DEFAULT_OUTPUT_DIR`, `DEFAULT_CUSHION_DIR`

### Supply your own paths

Convert to PNG previews:

```bash
dat-png --input-dir path/to/input --output-dir path/to/output
```

Convert to `.sxm`:

```bash
dat-sxm --input-dir path/to/input --output-dir path/to/output
```

If your cushion files are stored somewhere else, provide them explicitly:

```bash
dat-sxm --input-dir path/to/input --output-dir path/to/output --cushion-dir path/to/file_cushions
```

The cushion paths exist as particular byte sequences that all nanonis .sxm images have. they are provided in `\src\file_cushions`. The cushion path is inserted default relative to the repo root so you shouldnt need to edit any default file paths with respect to those. 


## Repository Contents

- [nanonis_tools](nanonis_tools): installable converter source code
- [scripts](scripts): thin wrappers for direct script execution
- [src/file_cushions](src/file_cushions): required layout assets for `.sxm` generation
- [data/sample_input](data/sample_input): two small example `.dat` files

## Notes

- `dat-sxm` writes output files using the input filename stem.
- The current `.sxm` timestamp parsing expects filenames of the form `AyyMMdd.HHmmss.dat`.
