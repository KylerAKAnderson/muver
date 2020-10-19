from itertools import repeat
from multiprocessing import Pool
from io import StringIO
from importlib import import_module
import os
import sys
import time
import pickle
import re

from . import bias_distribution as bias_dist
from . import depth_distribution as depth_dist
from . import read_processing as read_processing
from . import reference as reference
from .repeat_indels import fit_repeat_indel_rates, read_fits
from .repeats import create_repeat_file
from .sample import (Sample, read_samples_from_text,
                    generate_experiment_directory, write_sample_info_file)
from .utils import read_repeats, read_config
from .variant_list import VariantList
from . import wrappers
from .wrappers import bowtie2, gatk, picard, samtools
from .depth import describe_regions, correct_depths

def process_sams(args):
    '''
    Process SAM files for a given sample.  Perform the following:
    
    - Remove read pairs on different chromosomes.
    - Filter by MAPQ value.
    - Add read groups based on sample name.
    - Deduplicate.
    - Realign indels.
    - Fix mate information.
    - Merge processed BAMS for a given sample.
    '''
    sample_name, intermediate_files, reference_assembly, max_records = args
    
    report = StringIO("")
    log = SimpleLogger(where=report, level=1)
    
    log('- ' + sample_name, '\n')
    for i in range(len(intermediate_files['_sams'])):
        log('- - sam #' + str(i), '\n')
        
        log('- - - Removing Discordant Pairs')
        read_processing.remove_diff_chr_pairs(
            intermediate_files['_sams'][i],
            intermediate_files['_same_chr_sams'][i],
        )
        log.check()
        log('- - - Mapping Quality Filtering (samtools)')
        samtools.mapq_filter(
            intermediate_files['_same_chr_sams'][i],
            intermediate_files['_mapq_filtered_sams'][i],
        )
        log.check()
        log('- - - Regrouping (picard)')
        picard.add_read_groups(
            intermediate_files['_mapq_filtered_sams'][i],
            intermediate_files['_read_group_bams'][i],
            sample_name,
            intermediate_files['tmp_dirs'][i],
            max_records,
        )
        log.check()
        log('- - - Deduplicating (picard)')
        picard.deduplicate(
            intermediate_files['_read_group_bams'][i],
            intermediate_files['_deduplicated_bams'][i],
            intermediate_files['_deduplication_metrics'][i],
            intermediate_files['tmp_dirs'][i],
            max_records,
        )
        log.check()
        log('- - - Finding Realignment Targets (gatk)')
        gatk.realigner_target_creator(
            reference_assembly,
            intermediate_files['_deduplicated_bams'][i],
            intermediate_files['_interval_files'][i],
        )
        log.check()
        log('- - - Realigning for Indels (gatk)')
        gatk.indel_realigner(
            reference_assembly,
            intermediate_files['realignment_logs'][i],
            intermediate_files['_deduplicated_bams'][i],
            intermediate_files['_interval_files'][i],
            intermediate_files['_realigned_bams'][i],
        )
        log.check()
        log('- - - Fixing Mate Information (picard)')
        picard.fix_mate_information(
            intermediate_files['_realigned_bams'][i],
            intermediate_files['_fixed_mates_bams'][i],
            intermediate_files['tmp_dirs'][i],
            max_records,
        )
        log.check()

    log('- - Merging Bams (samtools)')
    samtools.merge_bams(
        intermediate_files['_fixed_mates_bams'],
        intermediate_files['merged_bam'],
    )
    log.check()
    return report.getvalue()

def characterize_repeat_indel_rates(args):
    '''
    For a given sample, fit repeat indel rates.
    '''
    intermediate_files, repeat_file, repeat_indel_header = args

    if os.path.isfile(repeat_file + '.sample'):
        repeats = read_repeats(repeat_file + '.sample')
    else:
        repeats = read_repeats(repeat_file)
    
    fit_repeat_indel_rates(
        repeats,
        intermediate_files['merged_bam'],
        intermediate_files['repeat_indel_fits'],
        repeat_indel_header,
    )

def analyze_depth_distribution(args):
    '''
    For a given sample, analyze read depths.  Perform the following:

    - Fit strand bias values to a log-normal distribution.
    - Fit depth values to a normal distribution.
    - Filter genome positions observing the normal distribution of depths.

    Return the passed index and the standard deviation of the log-normal
    distribution fit to strand bias values.
    '''
    index, sample_name, intermediate_files,\
      ploidy, cnv_regions, details = args

    reference_assembly, experiment_directory, chrom_sizes,\
      old_filter, dcmodule, onstage, tostage, options = details
    
    report = StringIO("")
    log = SimpleLogger(where=report, level=1)
    
    def nextStage():
        nonlocal onstage
        onstage += 10
        return onstage//10 >= tostage//10
    
    log('- ' + sample_name, '\n')
    
    if nextStage():
        log('- - Running SAMTools mpileup ({})'.format(onstage))
        samtools.run_mpileup(
            intermediate_files['merged_bam'],
            reference_assembly,
            intermediate_files['_mpileup_out'],
        )
        log.check()
    
    if nextStage():
        log('- - Calculating Strand Bias Dist. ({})'.format(onstage))
        _, strand_bias_std = bias_dist.calculate_bias_distribution_mpileup(
            intermediate_files['_mpileup_out'],
            intermediate_files['strand_bias_distribution'],
        )
        log.check()
    
    depths = depth_dist.read_raw_depths_mpileup(
        intermediate_files['_mpileup_out'],
        chrom_sizes,
    )
    
    if nextStage():
        log('- - Writing Bedgraph ({})'.format(onstage))
        depth_dist.write_bedgraph(
            depths, 
            intermediate_files['depth_bedgraph']
        )
        log.check()
    
    if dcmodule and nextStage():
        if dcmodule == 'kunkel':
            log('- - Correcting Depths ({})'.format(onstage))
            depths = correct_depths(
                depths,
                sample_name,
                ploidy,
                experiment_directory,
                options
            )
            log.check()
    
        else:
            log('- - Correcting Depths w/ {} ({})'.format(dcmodule, onstage))
            sys.path.append(dcmodule)
            dcmodule = import_module(dcmodule)
            depths = dcmodule.correct_depths(
                depths,
                sample_name,
                ploidy,
                experiment_directory,
                options
            )
            log.check()
    
        log('- - Writing Corrected Bedgraph')
        depth_dist.write_bedgraph(
            depths, 
            intermediate_files['corrected_bedgraph']
        )
        log.check()
    
    if nextStage():
        if old_filter:
            depth_dist.convert_to_percopy(
                depths, 
                ploidy, 
                cnv_regions,
            )
            mu, sigma = depth_dist.calculate_depth_distribution_dict(
                depths,
                intermediate_files['depth_distribution'],
            )
            log.check()

            log('- - Filtering by Depth ({})'.format(onstage))
            depth_dist.filter_regions_by_depth(
                depths,
                mu,
                sigma,
                intermediate_files['filtered_sites'],
            )
            log.check()

            cnvbg = None
        else:
            cnvbg = os.path.join(
                experiment_directory,
                'cnv_bedgraphs',
                sample_name + '.CNVs.bedGraph'
            )
            log('- - Characterizing Depth Regions ({})'.format(onstage))
            describe_regions(
                depths,
                ploidy,
                sample_name,
                cnvbg,
                intermediate_files['filtered_sites'],
                intermediate_files['depth_distribution'],
                experiment_directory,
                options
            )
            log.check()

    return index, strand_bias_std, cnvbg, report.getvalue()

def run_pipeline(reference_assembly, fastq_list, control_sample,
                 experiment_directory, p=1, excluded_regions=None,
                 fwer=0.01, max_records=1000000, clear_tmps=True, 
                 dcmodule='kunkel', old_filter=False, stage=0):
    
    '''
    Sample preparation wrapper to allow for serialization
    and temporary file management. Passes necessary information to the full _run_pipeline
    '''
    
    # Sample-independent preparation
    if dcmodule is 'None': dcmodule = None
    if dcmodule and dcmodule is not 'kunkel':
        try:
            import_module(dcmodule)
        except:
            if not os.path.exists(dcmodule):
                sys.stderr.write(
                    ('Cannot locate depth correction '\
                     'module: {}.').format(dcmodule))
            else:
                sys.stderr.write(
                    '{} was not able to be imported.'.format(dcmodule))
            return
    
    repeat_file = '{}.repeats'.format(os.path.splitext(reference_assembly)[0])
    uconfig_file = '{}.cfg'.format(os.path.splitext(reference_assembly)[0])
    options = dict()
    
    if not reference.check_reference_indices(reference_assembly):
        sys.stderr.write('Reference assembly not indexed. Run '
            '"muver index_reference".\n')
        return
    if not os.path.exists(repeat_file):
        sys.stderr.write('Repeats not found for reference assembly. Run '
            '"muver create_repeat_file".\n')
        return
    if os.path.exists(uconfig_file):
        with open(uconfig_file, 'r') as IN:
            content = IN.read()
            options.update(read_config(content))
    
    success = False
    try:
        # Sample preparation
        samples, opt2 = read_samples_from_text(
            fastq_list, exp_dir=experiment_directory)
        options.update(opt2)
        
        if not stage:
            generate_experiment_directory(experiment_directory)
            for sample in samples:
                sample.generate_intermediate_files()
        else:
            frompickle = os.path.join(experiment_directory, 'MuverPickle')
            if not os.path.exists(frompickle):
                sys.stderr.write(
                   ('No intermittent run found. '\
                    'Make sure {} is correctly placed, ' \
                    'or run Muver at least once with ' \
                    '--clear_tmps=N and --stage=0 (default).').format(frompickle))
                return
            with open(frompickle, 'rb') as IN:
                samples = pickle.load(IN)
        
        control_sample = next(
                (x for x in samples if x.sample_name == control_sample), None)
        
        # Hand off for processing
        _run_pipeline(reference_assembly, samples, control_sample,
                 experiment_directory, options, p, excluded_regions,
                 fwer, max_records, dcmodule, old_filter, stage)
        
        success = True
    finally:
        if clear_tmps in ['Y', 'Yes'] or \
           ((clear_tmps in ['S', 'On_Success']) and success):
            for sample in samples:
                sample.clear_temp_files()
                sample.clear_temp_file_indices()
        else:
            pickleto = os.path.join(experiment_directory, 'MuverPickle')
            if not os.path.exists(experiment_directory):
                return
            with open(pickleto, 'wb') as OUT:
                pickle.dump(samples, OUT)

def _run_pipeline(reference_assembly, samples, control_sample,
                 experiment_directory, options, p=1, excluded_regions=None,
                 fwer=0.01, max_records=1000000, dcmodule=None,
                 old_filter=False, tostage=0):
    '''
    Run the MuVer pipeline considering input FASTQ files. All files written
    to the experiment directory.
    '''
    
    # Preamble for logging and state
    t = time.strftime('%c')
    stamp = "\n{0}\n== On {1} ==\n{0}\n".format("="*(9+len(t)), t)
    
    logname = 'muver_p0-{}_{}.txt'.format(
        control_sample.sample_name,
        re.sub('[ |:]','',t)
    )
    logfile = open(os.path.join(experiment_directory, logname), 'a+')
    log = SimpleLogger(logfile)
    log.raw(stamp)
    
    wrappers.experiment_directory = experiment_directory
    with open(os.path.join(experiment_directory, 'muver_externals_error.txt'), 'w') as err,\
         open(os.path.join(experiment_directory, 'muver_externals_output.txt'), 'w') as output:
        err.write(stamp)
        output.write(stamp)
    
    pool = Pool(p)
    
    onstage = 0
    def nextStage():
        nonlocal onstage
        onstage += 100
        return onstage//100 >= tostage//100
    
    # Pipeline Stages
    
    if nextStage():
        log('Aligning w/ Bowtie2 ({})'.format(onstage), end='\n')
        for sample in samples:
            for i, fastqs in enumerate(sample.fastqs):
                if len(fastqs) == 2:
                    f1, f2 = fastqs
                else:
                    f1 = fastqs[0]
                    f2 = None
                log('- ' + sample.sample_name)
                bowtie2.align(f1, reference_assembly, sample._sams[i],
                              fastq_2=f2, p=p)
                log.check()
    
    if nextStage():
        log('Processing SAMs ({})'.format(onstage), '\n')
        reports = pool.map(process_sams, zip(
            [s.sample_name for s in samples],
            [s.get_intermediate_file_names() for s in samples],
            repeat(reference_assembly),
            repeat(max_records),
        ))
        for report in reports:
            log.raw(report)
        log.timer.lap()
    
    haplotype_caller_vcf = os.path.join(
        experiment_directory,
        'gatk_output',
        'haplotype_caller_output.vcf'
    )
    haplotype_caller_log = os.path.join(
        experiment_directory,
        'logs',
        'haplotype_caller.log'
    )
    
    if nextStage():
        log('Calling Haplotypes w/ GATK ({})'.format(onstage))
        
        bams = [s.merged_bam for s in samples]
        gatk.run_haplotype_caller(
            bams,
            reference_assembly,
            haplotype_caller_vcf,
            haplotype_caller_log,
            nct=p,
        )
        log.check()

    chrom_sizes = reference.read_chrom_sizes(reference_assembly)
    
    if nextStage():
        details = [reference_assembly, experiment_directory, chrom_sizes,
               old_filter, dcmodule, onstage, tostage, options]
        log('Analyzing Depth ({})'.format(onstage), '\n')
        results = pool.map(analyze_depth_distribution, zip(
            range(len(samples)),
            [s.sample_name for s in samples],
            [s.get_intermediate_file_names() for s in samples],
            [s.ploidy for s in samples],
            [s.cnv_regions for s in samples],
            repeat(details),
        ))
        for index, strand_bias_std, cnvbg, report in results:
            samples[index].strand_bias_std = strand_bias_std
            if cnvbg:
                samples[index].cnv_bedgraph = cnvbg
                samples[index].cnv_regions = samples[index].read_cnv_bedgraph()
            log.raw(report)  
        log.timer.lap()
    
    repeat_file = '{}.repeats'.format(os.path.splitext(reference_assembly)[0])
    
    if nextStage():
        log('Characterize Repeats ({})'.format(onstage))
        pool.map(characterize_repeat_indel_rates, zip(
            [s.get_intermediate_file_names() for s in samples],
            repeat(repeat_file),
            [s.repeat_indel_header for s in samples],
        ))
        for sample in samples:
            sample.repeat_indel_fits_dict = read_fits(sample.repeat_indel_fits)
        log.check()
    
    
    if nextStage():
        log('Calling Variants ({})'.format(onstage))
        variants = VariantList(
            haplotype_caller_vcf, samples, excluded_regions, repeat_file,
            control_sample, chrom_sizes, fwer)
        log.check()
    
        text_output = os.path.join(
            experiment_directory,
            'output',
            'mutations.txt'
        )
        vcf_output = os.path.join(
            experiment_directory,
            'output',
            'mutations.vcf'
        )
        variants.write_output_table(text_output)
        variants.write_output_vcf(vcf_output)

        sample_info_file = os.path.join(
            experiment_directory, 'sample_info.txt')
        write_sample_info_file(samples, sample_info_file)
    
    log('Done :)')
    logfile.close()

class SimpleLogger():
    
    def __init__(self, where=sys.stdout, level=0):
        self.timer = Timer()
        self.where = where
        self.level = level
    
    def check(self):
        print(u' [\u2714]', 'in {:.3f}s'.format(self.timer.lap()), file=self.where)
    
    def raw(self, msg, end=''):
        print(msg, file=self.where, end=end)
    
    def __call__(self, msg, end=''):
        fillign = 45 - 2*self.level
        print('> '*self.level + self.timer.Total(), ('{:<'+str(fillign)+'}').format(msg), end=end, file=self.where)

class Timer():
    
    def __init__(self):
        time.perf_counter()
        self.t0 = time.perf_counter()
        self.ti = self.t0
    
    ''' returns total time elapsed'''
    def total(self):
        return time.perf_counter() - self.t0
    
    def Total(self):
        return form(self.total())
    
    ''' returns time since last lap'''
    def lap(self):
        tf = time.perf_counter()
        dt = tf - self.ti
        self.ti = tf
        return dt
    
    def Lap(self):
        return form(self.lap())
    
def form(t):
    return '{:02d}:{:02d}:{:06.3f}'.format(
        int(t//60**2),
        int(t//60%60),
        t%60)

if __name__ == '__main__':
    run_pipeline(*sys.argv[1:], p=8, max_records=300000)
    
    