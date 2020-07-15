# MuVer
MuVer is an analytical framework developed to improve sensitivity and increase accuracy in mutation identification from high-throughput sequencing data. MuVer provides significantly increased accuracy in challenging genomic contexts, including low complexity repetitive sequences. The MuVer framework has been applied to data from mutation accumulation experiments in yeast. 

## Requirements
The latest version of MuVer was developed using Python 3.7.6. In addition to requirements specified in setup.py, MuVer requires installation of the following tools:
* [Bowtie2](http://bowtie-bio.sourceforge.net/bowtie2/index.shtml)
* [GATK, version 3.7-0](https://software.broadinstitute.org/gatk/download/)
* [picard](https://broadinstitute.github.io/picard/)
* [samtools](http://www.htslib.org/download/)

## Installation
Proper function of MuVer requires the paths to depencies to be set.  To do this, manually set the paths in `paths.cfg` using a text editor.

After the correct paths have been set, install MuVer with the following command:
```
python setup.py install
```
A [Docker image](https://hub.docker.com/r/lavenderca/muver/) is also available.

## Usage
To run, a copy of 'options.cfg' may be needed in the working directory.
Read 'options.cfg' for further setup instructions and see the `--dcmodule` option under `run_pipeline` in the [manual](docs/manual.md#run_pipeline) for information.

All of MuVer's functions may be accessed using its command line interface. General usage is as follows:
```
muver COMMAND [OPTIONS] [ARGS]...
```
A list of commands can be found by using the following:
```
muver --help
```
Details about each command can be found by using the following:
```
muver COMMAND --help
```
See the [manual](docs/manual.md) for further usage details.

## Authors
MuVer was conceptualized by Scott Lujan and Adam Burkholder.

MuVer was written by Adam Burkholder and Christopher Lavender.

MuVer is maintained by Kyler Anderson and Adam Burkholder.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments
This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.
