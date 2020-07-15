# muver
muver is an analytical framework developed to improve sensitivity and increase accuracy in mutation identification from high-throughput sequencing data. muver provides significantly increased accuracy in challenging genomic contexts, including low complexity repetitive sequences. The muver framework has been applied to data from mutation accumulation experiments in yeast.

## Requirements
muver was developed using Python 3.7.6. In addition to requirements specified in setup.py, muver requires installation of the following tools:
* [Bowtie2](http://bowtie-bio.sourceforge.net/bowtie2/index.shtml)
* [GATK, version 3.7-0](https://software.broadinstitute.org/gatk/download/)
* [picard](https://broadinstitute.github.io/picard/)
* [samtools](http://www.htslib.org/download/)

## Installation
After download, nagivate to the muver directory. Proper function of muver requires the paths to depencies to be set.  To do this, manually set the paths in `paths.cfg` using a text editor.

After the correct paths have been set, install muver with the following command:

```
python setup.py install
```

## Functions
All of muvers functions may be accessed using its command line interface. General usage is as follows:

```
muver COMMAND [OPTIONS] [ARGS]...
```
Available commands:
* [run_pipeline](#run_pipeline)
* [call_mutations](#call_mutations)
* [calculate_bias_distribution](#calculate_bias_distribution)
* [calculate_depth_distribution](#calculate_depth_distribution)
* [calculate_depth_ratios](#calculate_depth_ratios)
* [calculate_read_depths](#calculate_read_depths)
* [correct_depths](#correct_depths)
* [create_repeat_file](#create_repeat_file)
* [extract_repeat_file_sample](#extract_repeat_file_sample)
* [fit_repeat_indel_rates](#fit_repeat_indel_rates)
* [index_reference](#index_reference)
* [plot_allelic_fraction](#plot_allelic_fraction)

### run_pipeline
```
muver run_pipeline [OPTIONS] REFERENCE_ASSEMBLY FASTQ_LIST CONTROL_SAMPLE_NAME EXPERIMENT_DIRECTORY
```

Runs the full muver pipeline, from FASTQs to called genotypes and mutations. For details about the mutation calling methodology, see the companion manuscript.

Input samples and parameters are specified in the `FASTQ_LIST` file. In this file, each row corresponds to an individual sample. The row contents in turn are specified by header fields in the first row. For a complete example, see example_fastq_list.txt.

The default depth correction is intended for use by the lab developing MuVer. *To turn off depth correction, add the option --dcmodule=''.* For depth profiles that are already sufficiently flat (ignoring potential CNVs) the depth correction will have minimal effect, but may cause artifacts.

The depth submodule refers to an options.cfg file *in the working directory* for tweaking parameters of its procedures. *If the default depth correction is used, this file is* ***NOT*** *optional.* This is because the default depth correction assumes a yeast genome and requires information about chromosome number in order to operate. More information is provided in options.cfg.

Prior to running, the reference assembly must be indexed and repeats called. To do this, run the muver functions [index_reference](#index_reference) and [create_repeat_file](#create_repeat_file).

#### Output format
Output files are placed into the `EXPERIMENT_DIRECTORY`. When `run_pipeline` is started, the following file structure is automatically generated in `EXPERIMENT_DIRECTORY`:
```
EXPERIMENT_DIRECTORY/
|   sample_info.txt
|-- bams/
|-- cnv_bedgraphs/
|-- corrected_depth_distributions/
|-- depth_distributions/
|-- filtered_sites/
|-- fits/
|-- gatk_output/
|-- log/
|-- output/
```
The sample_info.txt file contains parameters and paths to output files associated with the run. To tweak the parameters associated with analysis, the sample_info.txt file may be copied and modified; the modified file may then be used as an argument for [call_mutations](#call_mutations). Details for each additional output file can found in documentation for the associated function. The contents of each directory are described below:
* bams/

  Contains the final processed BAM alignment files. Alignment files are processed through the following steps:
  1.  Remove pairs mapping on different chromosomes.
  2.  Filter by mapping quality.
  3.  Add read groups by sample.
  4.  Deduplicate.
  5.  Realign indels (GATK).

* cnv_bedgraphs/
  
  Contains depth based CNV calls made by the depth submodule. Each CNV described by a line in bedGraph format as folllows: chromosome start end ploidy.

* corrected_depth_distributions/

  Contains bedGraph files describing read depths in processed BAM after passing the depths through a correction process for the sake of improving CNV calls. If --dcmodule='' is specified when using [run_pipeline](#run_pipeline), this directory will not be populated.

  *Please note that [correct_depths](#correct_depths) still uses v0.1.0 depth correction and does not ouput to this directory. This behavior will be addressed in v0.2.1.*

* depth_distributions/

  Contains depth and strand bias distributions. In addition, contains bedGraph files describing read depths in processed BAM files. Depth and strand bias distributions are generated using [calculate_depth_distribution](#calculate_depth_distribution) and [calculate_bias_distribution](#calculate_bias_distribution), respectively. Read depths are calculated using [calculate_read_depths](#calculate_read_depths).

  *Please note that [calculate_depth_distribtion](#calculate_depth_distribution) still produces the parameters from v0.1.0 depth distributions. This inconsistency will be addressed by v0.2.1.*

* filtered_sites/

  Contains regions filtered on the basis of abnormal read depth. 

  *Please note that [calculate_depth_distribution](#calculate_depth_distribution) still uses v0.1.0 filtering techniques relying on v0.1.0 depth distribution parameters as produced by [calculate_depth_distribtion](#calculate_depth_distribution). This inconsistency will be addressed by v0.2.1

* fits/

  Contains details of the fitting of observed indel rates to logistic distributions. These files are generated using [fit_repeat_indel_rates](#fit_repeat_indel_rates).
* gatk_output/

  Contains files generated by GATK. A VCF file from GATK HaplotypeCaller is required to call mutations.
* logs/

  Contains log files from GATK processes.
* output/

  Contains final called mutations in both table and VCF formats. These files are generated using [call_mutations](#call_mutations).

#### Options
* `-p, --processes INTEGER`

  Number of processes/threads to allocate to muver and its dependencies (default 1)

* `--excluded_regions PATH`

  Path to a BED file specifying genomic regions to ignore during mutation calling.

* `--fwer FLOAT`

  Rate at which to control family-wise error.

* `--max_records`

  Maximum number of reads to store in memory when sorting BAM files (default = 1,000,000). This value is passed to picard as the parameter 'MAX_RECORDS_IN_RAM', and requires approximately 2 GB of memory per million, multiplied by the count of samples or the number of simultaneous processes/threads specified with '-p', whichever is lower. This value should be modified when any input FASTQ pair contains more than 1 billion reads, to prevent picard from opening more temporary files than the system will allow. This file limit is typically 1024, and can be queried by running the following command: 'ulimit -n'. This parameter should be set no lower than (Maximum Read Count / Limit). In some cases, the limit may be increased to allow reduced memory usage. The maximum value may be queried using 'ulimit -Hn'.

* `--clear_tmps`

  Whether to clear temporary files and serialize the sample information after running the pipe_line. Valid options include 'Yes' (or 'Y'), 'On_Success' (or 'S', this is the Default), and 'No' (or 'N'). '(Y)es' will delete any temporary files even if the run ends in an error whereas 'On_(S)uccess' only deletes them if the run ends successfully. '(N)o' will always leave behind all temporary files.

This behavior is to be used in conjunction with the `--stage` option to resume the pipeline from intermediate stages for manual interventions or interrupted runs.

* `--old_filter`

  A True or False value indicating whether to use v0.1.0 depth distribution calculations and depth filtering techniques.

* `--dcmodule`

  The name of an alternative module to use for correcting the depth profiles in preparation for CNV calling. *By default, a correction procedure somewhat specific to the lab developing MuVer is used.* To turn off all depth correction, pass --dcmodule=''.

  To provide custom depth correction, specify the module name and make sure it is importable from the working directory. The module must have a correct_depths function that takes as arguments (in the given order): a dictionary of the depth profiles (by track name of the `REFERENCE_ASSEMBLY`), the sample name, the ploidy (as given in `FASTQ_LIST`), and an output folder used to convey the experiment directory.

* `--stage`

  Allows the pipeline to begin at intermediate steps. Run a set of samples without this option before attempting to use it. Temporary files and sample information may be serialized (see the `--clear_tmps` option) for a given run. Providing a non-zero value for the stage instructs MuVer to load the serialized information and use the previous temporary files. Follow the integer codes next to the names of stages in the ouput to indicate the desired entry point. Starting later than the last successful stage produces behavior that is left undefined and is therefore unsupported.

#### Arguments
* `REFERENCE_ASSEMBLY`

  Path to the reference assembly in FASTA format. Run the muver functions [index_reference](#index_reference) and [create_repeat_file](#create_repeat_file) prior to `run_pipeline`.
* `FASTQ_LIST`

  Path to a tab-delimited text file that describes files and parameters for each sample.
* `CONTROL_SAMPLE_NAME`

  Name of the control ancestor sample. Must be consistent with the name specified in the `FASTQ_LIST`.
* `EXPERIMENT_DIRECTORY`

  Path to the experiment directory, where output and intermediate files will be written.

#### FASTQ_LIST fields
* "Sample Name"

  Required. Name of the sample.
* "Mate 1 FASTQ"

  Required. Path to the mate 1 FASTQ file. If multiple FASTQ files are used, enter in a comma-separated list.
* "Mate 2 FASTQ"

  Path to the mate 2 FASTQ file. If multiple FASTQ files are used, enter in a comma-separted list consistent with mate 1.
* "Ploidy"

  Sets the base ploidy for the sample. Individual positions will be superceded by the CNV bedGraph. If not specified, a sample will be assumed to be diploid (ploidy = 2).
* "CNV bedGraph"

  Path to a bedGraph file with ploidy values for individual regions. If the value of a region is '0', no mutations will be called in that region.

### call_mutations
```
muver call_mutations [OPTIONS] REFERENCE_ASSEMBLY CONTROL_SAMPLE_NAME SAMPLE_LIST INPUT_VCF OUTPUT_HEADER
```
Calls mutations from a pre-computed GATK HaplotypeCaller VCF file.

Prior to running, the reference assembly must be indexed and repeats called. To do this, run the muver functions `index_reference` and `create_repeat_file`.

#### Output format
`call_mutations` generates two files describing the called mutations: a tab-delimited table (\*.txt) and a VCF file (\*.vcf).

The tab-delimited table contains the following fields for each mutation:

* "Chromosome"
* "Position"

  Refers to the left-most position of the reference allele.
* "Excluded Region"

  Indicates if the variant falls within a user-defined excluded region.
* "Alleles"

  Alleles detected at that position separated by commas. The first allele is the reference allele.
* "Intersects Repeat"

  Indicates if the variant falls within a called repeat (see [create_repeat_file](#create_repeat_file)).
* "Repeat Correction Applied"

  Indicates if a repeat correction was applied during genotype calling.


* "CONTROL_SAMPLE Depths"

  Per-allele, per-strand read depths for all alleles. For instance, the first two values correspond to the forward and reverse strand read counts of the first listed allele.
* "CONTROL_SAMPLE Allele Count Flag"

  Indicates if at least two per-allele, per-strand depth values are greater than zero.
* "CONTROL_SAMPLE Depth Flag"

  Indicates if the sample has total read depth greater than or equal to a threshold of 20 reads.
* "CONTROL_SAMPLE Filter Flag"

  Indicates if the variant position intersects the filtered regions for the sample.
* "CONTROL_SAMPLE Ploidy"

  Ploidy for the sample at this position.
* "CONTROL_SAMPLE Genotype"

  Called genotype for the sample.
* "CONTROL_SAMPLE Subclonal Genotype"

  Called sub-clonal genotype for the sample.
* "CONTROL_SAMPLE Subclonal Frequency"

  Frequency of the called sub-clonal.
* "CONTROL_SAMPLE Genotyping Score"

  Score associated with the called genotype.

The following additional fields are included for each non-control sample:

* "SAMPLE Depths"

  Per-allele, per-strand read depths for all alleles. For instance, the first two values correspond to the forward and reverse strand read counts of the first listed allele.
* "SAMPLE Allele Count Flag"

  Indicates if at least two per-allele, per-strand depth values are greater than zero.
* "SAMPLE Depth Flag"

  Indicates if the sample has total read depth greater than or equal to a threshold of 20 reads.
* "SAMPLE Filter Flag"

  Indicates if the variant position intersects the filtered regions for the sample.
* "SAMPLE Composite Score"

  Composite score for mutation calling. Made relative to the control sample.
* "SAMPLE Read Difference Flag"

  Indicates if the called mutations pass thresholds. Considers both composite score and individual P values.
* "SAMPLE Ploidy"

  Ploidy for the sample at this position.
* "SAMPLE Genotype"

  Called genotype for the sample.
* "SAMPLE Subclonal Genotype"

  Called sub-clonal genotype for the sample.
* "SAMPLE Subclonal Frequency"

  Frequency of the called sub-clonal.
* "SAMPLE Genotyping Score"

  Score associated with the called genotype.
* "SAMPLE Mutations"

  Comma-delimited list of called mutations.

  Mutations are named according to [HGVS](http://varnomen.hgvs.org/) specifications. To address mutations not explicitly described in those specifications, the following conventions are used:

  * To describe copy-number variations, mutation names are formatted as follows: g.{POS}[gain/loss]{ALLELE}. For instance, g.1000gainAT describes a gain of the AT allele at genomic position 1000. The position reported is the left-most position of the associated reference allele.
  * If a non-reference allele is mutated, the position is given as the flanking nucleotides of the reference allele. For instance, if at genomic position 100 an A to C SNP was called where the reference allele was G, the following name would be used: g.99_101A>C.
  * Mutations are only identified as 'ins' or 'del' if the reference allele is modified. Otherwise, '>' notation is used. For instance, if a AT to A mutation was called where the reference allele was G at genomic position 100, the following name would be used: g.99_101AT>A

* "SAMPLE PAC Flag"

  Identifies a conversion event from one allele to another. Reported in a comma-delimited list with each entry corresponding to a called mutation.

The VCF file contains the following INFO fields:

* IntersectsRepeat

  Indicates if the variant falls within a called repeat (see [create_repeat_file](#create_repeat_file)).
* RepeatCorrectionApplied

  Indicates if a repeat correction was applied during genotype calling.

The VCF file contains the following FILTER fields:

* ExcRegion

  Indicates if the variant falls within a user-defined excluded region.

The VCF file contains the following FORMAT fields for each sample:

* SAC: Strand allele counts

  Per-allele, per-strand read depths for all alleles. For instance, the first two values correspond to the forward and reverse strand read counts of the first listed allele.
* ACF: Allele count flag

  Indicates if at least two per-allele, per-strand depth values are greater than zero.
* DF: Depth flag

  Indicates if the sample has total read depth greater than or equal to a threshold of 20 reads.
* PD: Ploidy

  Ploidy for the sample at this position.
* GT: Genotype

  Called genotype for the sample.
* SGT: Subclonal Genotype

  Called sub-clonal genotype for the sample.
* SF: Subclonal frequency

  Frequency of the called sub-clonal.
* SV: Subclonal valid flag

  Indicates if validity tests for the subclonal allele pass thresholds.
* RD: Read difference flag

  Indicates if the called mutations pass thresholds. Considers both composite and individual P values.
* MT: Called mutations

  Comma-delimited list of called mutations.
* PAC: Called PAC flags

  Identifies a conversion event from one allele to another. Reported in a comma-delimited list with each entry corresponding to a called mutation.

#### Options
* `--excluded_regions PATH`

  Path to a BED file specifying genomic regions to ignore during mutation calling.

* `--fwer FLOAT`

  Rate at which to control family-wise error, default = 0.01.

#### Arguments
* `REFERENCE_ASSEMBLY`

  Path to the reference assembly in FASTA format. Run the muver functions [index_reference](#index_reference) and [create_repeat_file](#create_repeat_file) prior to `call_mutations`.
* `CONTROL_SAMPLE_NAME`

  Name of the control ancestor sample. Must be consistent with the name specified in the `SAMPLE_LIST`.
* `SAMPLE_LIST`

  Path to a tab-delimited text file that describes files and parameters for each sample.
* `INPUT_VCF`

  Path to a VCF from GATK HaplotypeCaller to be used for mutation calling.
* `OUTPUT_HEADER`

  Path to be used as a prefix in naming output files.

#### SAMPLE_LIST headers
* "Sample Name"

  Required. Name of the sample.
*  "Merged BAM"

  Required. BAM file for aligned reads. For processing guidelines, see [run_pipeline](#run_pipeline) for details.
* "Strand Bias Std"

  Required. Strandard deviation of the natural log-transformed ratio of per-strand counts.
* "Repeat Indel Fits"

  Required. Path to a repeat indel fits file. Describes the frequencies of indels based on repeat unit length and repeat tract length. Generated by [fit_repeat_indel_rates](#fit_repeat_indel_rates).
* "Ploidy"

  Sets the base ploidy for the sample. Individual positions will be superceded by the CNV bedGraph. If not specified, a sample will be assumed to be diploid (ploidy = 2).
* "CNV bedGraph"

  Path to a bedGraph file with ploidy values for individual regions. If the value of a region is '0', no mutations will be called in that region.
* "Filtered Sites"

  Path to a filtered sites text file. Mutations are not called at positions described in this file. Generated by [calculate_depth_distribution](#calculate_depth_distribution).

### calculate_bias_distribution
```
muver calculate_bias_distribution [OPTIONS] BAM_FILE REFERENCE_ASSEMBLY OUTPUT_BIAS_DISTRIBUTION
```
Calculates the strand-bias distribution for a BAM file. For each position in the genome, finds the ratio of per-strand count and takes the natural log.  These values are then compiled in a histrogram and fit to a normal distribution. The histrogram and the parameters of the fit are reported in the output file.

#### Output format
The first two lines of the output file give the parameters of the fit. The remainder of the file gives a histogram of observed frequencies and the corresponding fit value.

#### Arguments
* `BAM_FILE`

  Path to the input BAM alignment file.
* `REFERENCE_ASSEMBLY`

  Path to the reference assembly in FASTA format.
* `OUTPUT_BIAS_DISTRIBUTION`

  Path to the output text file describing the distribution of natural log-transformed per strand ratios.

### calculate_depth_distribution
```
muver calculate_depth_distribution [OPTIONS] BEDGRAPH_FILE REFERENCE_ASSEMBLY OUTPUT_DEPTH_DISTRIBUTION
```
***Version 0.1.0 Behavior***

Calculates the distribution of values per chromosome copy for a bedGraph file. Generates a histogram of values and fits those values to a normal distribution. The histogram and the parameters of the fit are reported in the output file.

#### Output format
The first two lines of the output file give the parameters of the fit. The remainder of the file gives a histogram of observed frequencies and the corresponding fit value.

In the filtered regions BED file, each row corresponds to a filtered region.  Three values are given in each row: chromosome, interval start, and interval end.

#### Options
* `--output_filtered_regions TEXT`

  If specified, creates a list in BED format of regions to filter prior to mutation calling. Filters on the basis of abnormal depth relative to the distribution of values in the bedGraph file.
* `--ploidy INTEGER`

  Sample ploidy, default = 2.  This value is utilized as the assumed copy number in non-CNV regions.

* `--cnv_bedgraph_file TEXT`

  If provided, values contained in this file will be utilized rather than the global ploidy when muver determines depth per copy for a given position.  File is assumed to be bedGraph format, consisting of four tab-separated columns:  chromosome, interval start, interval end, copy number.

* `--p_threshold FLOAT`

  Threshold to be utilized in determination of abnormal depth, default = 1 x 10^-4.

* `--merge_window INTEGER`

  Adjacent positions with abnormal depth, falling on the same side of the genomic median value, within this window (default = 1000) will be used to define the edges filtered regions.

#### Arguments
* `BEDGRAPH_FILE`

  Path to a bedGraph file describing per-base depths.
* `REFERENCE_ASSEMBLY`

  Path to reference assembly in FASTA format.
* `OUTPUT_DEPTH_DISTRIBUTION`

  Path to the output text file describing the distribution of depth values.

### calculate_depth_ratios
```
muver calculate_depth_ratios [OPTIONS] BEDGRAPH_FILE REFERENCE_ASSEMBLY OUTPUT_FILE
```
For each position in the genome, finds the ratio of the depth at that position to the genome-wide mean. These ratios are then binned by the shorter distance to the end of a chromosome, and the median values of these bins are reported. Designed to aid in finding values for per-sample depth correction.

#### Output format
The output file considers values binned based on relative distances from chromosome ends, with each row corresponding to a given bin. The file has four fields: the bin start distance, the bin end distance, the center distance of the bin, and the median of binned values.

#### Options
* `--mean FLOAT`

  If specified, this value will be used as the mean. If not, the mean will automatically be calculated.

#### Arguments
* `BEDGRAPH_FILE`

  Path to the input bedGraph file describing per-base depths.
* `REFERENCE_ASSEMBLY`

  Path to reference assembly in FASTA format.
* `OUTPUT_FILE`

  Path to the output text file describing the mean ratios.

### calculate_read_depths
```
muver calculate_read_depths [OPTIONS] BAM_FILE REFERENCE_ASSEMBLY OUTPUT_BEDGRAPH
```
Considering an input BAM file, calculates per-position read depths and writes to a bedGraph file. Read depth values are found considering samtools mpileup values.

#### Output format
The output is in standard bedGraph format, with read depths reported in the fourth column.

#### Arguments
* `BAM_FILE`

  Path to the input BAM alignment file.
* `REFERENCE_ASSEMBLY`

  Path to the reference assembly in FASTA format.
* `OUTPUT_BEDGRAPH`

  Path to the output bedGraph file describing per-base depths.

### correct_depths
```
muver correct_depths [OPTIONS] Y_INT SCALAR MEAN_LOG SD_LOG SLOPE REFERENCE_ASSEMBLY INPUT_BEDGRAPH OUTPUT_BEDGRAPH
```
***Version 0.1.0 Behavior***

Corrects depth values for samples with relatively higher depth at chromosome ends. Depth correction is performed using the sum of a log-normal cumulative distribution function and a linear function. The correction function is given below, where x is the distance from the chromosome end:

correction_factor(x) = SCALAR \* (1 - [1/2 + 1/2 erf((ln(x) - MEAN_LOG)) / (sqrt(2) \* SD_LOG)]) + Y_INT + SLOPE \* x

#### Output format
The output is in standard bedGraph format, with corrected read depths reported in the fourth column. Corrected depths are floored to the nearest integer.

#### Arguments
* `Y_INT`, `SCALAR`, `MEAN_LOG`, `SD_LOG`, `SLOPE`

  Parameters for the correction function.
* `REFERENCE_ASSEMBLY`

  Path to the reference assembly in FASTA format.
* `INPUT_BEDGRAPH`

  Path to the input bedGraph file describing per-base depths.
* `OUTPUT_BEDGRAPH`

  Path to the output bedGraph file with corrected depth values.

### create_repeat_file
```
muver create_repeat_file [OPTIONS] FASTA_FILE OUTPUT_REPEAT_FILE
```
Considering an input FASTA file, finds all repeats, and writes to an output text file.

#### Output format
Each line in the output file corresponds to a single identified repeat.  Each line hsa six fields: the chromosome, the complete repeat sequence, the length of the repeat unit, the repeat unit sequence, the start position of the repeat sequence (exclusive), and the end position (inclusive).

#### Arguments
* `FASTA_FILE`

  Path to the input sequence file in FASTA format.
* `OUTPUT_REPEAT_FILE`

  Path to output text file describing discovered repeat sequences.

### extract_repeat_file_sample
```
muver extract_repeat_file_sample REPEAT_FILE SAMPLE_SIZE
```
Given a previously generated repeat file, extracts a random sample of loci of user-define size.  Helpful in reducing memory usage in the analysis of large genomes.

#### Output format
Identical to repeat file above.  The file name will be automatically generated based on the input file name, appending ".sample".  If such a file exists, it will be automatically utilized by `run_pipeline` during determination of indel error rates.

#### Arguments
* `REPEAT_FILE`

  Path to the previously created repeat file.
* `SAMPLE_SIZE`

  Size of sample to be extracted.  A sample of 5,000,000 repeat sites has been utlized effectively to assess indel error rates in a high-depth human data set.

### fit_repeat_indel_rates
```
muver fit_repeat_indel_rates [OPTIONS] BAM_FILE REPEATS_FILE OUTPUT_FITS_FILE
```
Considering a BAM input file and a list of repeats, calculates the frequencies of indels in repeat regions and fits those values to a logistic function. Fits are found for each event class (insertion or deletion) and repeated sequence length, and fit parameters are written to an output text file. The repeats file is generated using [create_repeat_file](#create_repeat_file).

#### Output format
The output fits file describes parameters used to fit a logistic distribution to observed indel frequencies in repeat regions. Indels where grouped by the class of the intersected repeat and the length of associated repeat unit. Each line in the file gives fit parameters for a given indel group.

Optional output plots show the observed indel rates against fitted values. An individual plot is generated for each indel group. In plots, the observed frequencies are displayed as black dots, and the fitted values are displayed as a blue line.

#### Options
* `--output_plot_header TEXT`

  If specifies, plots the calculated fits against observed values. Plots are written to PNG files with the header used as a prefix.

#### Arguments
* `BAM_FILE`

  Path to the input BAM alignment file.
* `REPEATS_FILE`

  Path to the input repeats file.
* `OUTPUT_FITS_FILE`

  Path to the output text file containing fit parameters.

### index_reference
```
muver index_reference [OPTIONS] REFERENCE_ASSEMBLY
```
For the specified reference assembly, write index files in place. Indices are written using Bowtie2, picard, and samtools.

#### Arguments
* `REFERENCE_ASSEMBLY`

  Path to the input sequence file in FASTA format.

### plot_allelic_fraction
```
muver plot_allelic_fraction [OPTIONS] BAM_FILE REFERENCE_ASSEMBLY OUTPUT_FILE
```
Considering an input alignment file, report a histogram of allelic fractions. Designed to determine overall ploidy for a sample.

#### Arguments
* `BAM_FILE`

  Path to the input BAM alignment file.
* `REFERENCE_ASSEMBLY`

  Path to the reference assembly in FASTA format.
* `OUTPUT_FILE`

  Path to the output histogram text file.
