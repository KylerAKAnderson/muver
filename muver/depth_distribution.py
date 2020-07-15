import numpy
from scipy.stats import norm
from scipy.optimize import curve_fit
from utils import read_cnv_bedgraph

from fitting import gaussian


def read_raw_depths_mpileup(mpileup_file, chrom_sizes):
    '''
    Reads mpileup raw coverage into a dictionary, an entry per chromosome.
    '''
    depths = dict()

    with open(mpileup_file) as f:
        for line in f:

            line_split = line.strip().split()

            chromosome, position, reference_base, coverage = line_split[:4]
            position = int(position)
            coverage = float(coverage)

            if chromosome not in depths:
                depths[chromosome] = numpy.zeros(chrom_sizes[chromosome], \
                    dtype=numpy.int32)

            if int(coverage) > 0:
                bases = line_split[4]
            else:
                bases = ''

            i = 0
            while i < len(bases):
                if bases[i] == '^':
                    i += 1
                elif bases[i] == '*':
                    coverage += 1
                i += 1
            
            depths[chromosome][position-1] = coverage
                    
    return depths


def write_bedgraph(depths_dict, output_bedgraph):
    
    with open(output_bedgraph, 'w') as OUT:
        for chromosome in depths_dict:
            depths = depths_dict[chromosome]

            idxs = numpy.nonzero(depths[:-1] != depths[1:])[0]

            breaks = numpy.concatenate(([1], idxs+2, [len(depths)+1]))
            values = numpy.concatenate((depths[idxs], [depths[-1]]))
            
            for i, value in enumerate(values):
                if not value: continue
                OUT.write('{}\t{}\t{}\t{}\n'.format(
                    chromosome, 
                    breaks[i],
                    breaks[i+1],
                    value))

                          
def convert_to_percopy(depths, ploidy, cnv_regions):
    '''
    Convert a dictionary of raw mpileup depths to coverage per copy.
    '''
    for chromosome in depths:
        for i in range(len(depths[chromosome])):
            if (chromosome, i+1) in cnv_regions:
                depths[chromosome][i] //= cnv_regions[(chromosome, i+1)]
            else: 
                depths[chromosome][i] //= ploidy


def calculate_depth_distribution(depths, output):
    '''
    For a list of depths, create a histogram of depth values. Then
    fit those values using a normal distribution. Return fit parameters. Output
    fit parameters and the histogram in a TXT file.
    '''
    depth_max = max(depths)
    hist = numpy.histogram(depths, bins=range(1, depth_max + 2), density=True)

    p0_mu, p0_sigma = norm.fit(depths)

    popt, pcov = curve_fit(gaussian, hist[1][:-1], hist[0], p0=[p0_mu, p0_sigma])
    mu, sigma = popt
    sigma = abs(sigma)

    # TODO: this is largely copied from calculate_bias_distribution
    with open(output, 'w') as OUT:

        OUT.write('Average depth per copy: {}\n'.format(str(mu)))
        OUT.write('Standard deviation of depths per copy: {}\n\n'.format(str(sigma)))

        OUT.write('Depth distribution:\n\n')
        OUT.write('\t'.join(['Depth', 'Frequency', 'Fit value']) +
            '\n')

        for hist_value, _bin in zip(hist[0], hist[1]):
            OUT.write('\t'.join((
                str(_bin),
                str(hist_value),
                str(norm.pdf(_bin, mu, sigma)),
            )) + '\n')

    return mu, sigma


def calculate_depth_distribution_dict(depths_dict, output):
    '''
    Combine all nonzero depths from a coverage per copy dictionary into a list
    '''
    depths = []
    for chromosome in depths_dict:
        idxs = numpy.nonzero(depths_dict[chromosome])[0]
        depths.extend(depths_dict[chromosome][idxs])
    return calculate_depth_distribution(depths, output)


def calculate_depth_distribution_bedgraph(in_bedgraph, output, ploidy=2,
                                          cnv_bedgraph_file=None):
    '''
    Read depths from a bedGraph file, determine coverage per copy,
    and pass to calculate_depth_distribution.
    '''
    if cnv_bedgraph_file is not None:
        cnv_regions = read_cnv_bedgraph(cnv_bedgraph_file)
    else:
        cnv_regions = dict()

    depths = []

    with open(in_bedgraph) as f:
        for line in f:

            chromosome, start, end, coverage = line.strip().split()
            for i in range(int(start), int(end)):
                if (chromosome, i) in cnv_regions:
                    depths.append(int(float(coverage) / \
                        cnv_regions[(chromosome, i)]))
                else:
                    depths.append(int(float(coverage) / ploidy))

    return calculate_depth_distribution(depths, output)


def calculate_depth_distribution_mpileup(input_mpileup, output, ploidy,
                                         cnv_regions):
    '''
    Read depths from a mpileup TXT file, determine coverage per copy,
    and pass to calculate_depth_distribution.
    '''
    depths = []

    with open(input_mpileup) as f:
        for line in f:

            line_split = line.strip().split()

            chromosome, position, reference_base, coverage = line_split[:4]
            position = int(position)
            coverage = float(coverage)
            if int(coverage) > 0:
                bases = line_split[4]
            else:
                bases = ''

            i = 0
            while i < len(bases):
                if bases[i] == '^':
                    i += 1
                elif bases[i] == '*':
                    coverage += 1
                i += 1

            if coverage > 0:
                if (chromosome, position) in cnv_regions:
                    depths.append(int(coverage / \
                        cnv_regions[(chromosome, position)]))
                else:
                    depths.append(int(coverage / ploidy))

    return calculate_depth_distribution(depths, output)


def process_chromosome_values(chromosome, chromosome_values, mu, sigma, OUT,
                              p_threshold=0.0001, merge_window=1000, window=51):
    '''
    Go over depth values for a given chromosome in an input list and write
    to a list of filtered positions if a position is less or greater than
    threshhold values derived the cummulative distribution function of a
    normal distribution.

    chromosome -- The name of the chromosome.  Used only in printing.
    chromosome_values -- list of chromosome depths at every position.
    mu -- Describes normal distribution, used to filter abnormal depths.
    sigma -- Describes normal distribution, used to filter abnormal depths.
    OUT -- File handle to write filtered positions with abnormal depths.
    window -- Window to smooth depth values.
    p_threshold -- Probability applied to the CDF of the normal distribution
                   to generate depth thresholds for filtering.
    '''
    def write_interval_to_filter(chromosome, start, end):
        OUT.write('{}\t{}\t{}\n'.format(
            chromosome,
            str(start),
            str(end + 1),
        ))

    d = int((window - 1) / 2)
    norm_dist = norm(mu, sigma)

    keep_threshold = [mu, mu]
    filter_threshold = [float('-inf'), float('inf')]

    first = float('inf')
    last = float('-inf')
    side = 0
    last_side = 0

    max = len(chromosome_values)

    for i in range(0, max):

        if i < d:
            window_start = 0
            window_end = i + d + 1
        elif i >= (max - d):
            window_start = i - d
            window_end = max
        else:
            window_start = i - d
            window_end = i + d + 1

        window_depth = numpy.mean(chromosome_values[window_start:window_end])

        if not (
            window_depth >= keep_threshold[0] and
            window_depth <= keep_threshold[1]
        ):
            if (
                window_depth <= filter_threshold[0] or
                window_depth >= filter_threshold[1]
            ):
                if window_depth < mu:
                    side = -1
                else:
                    side = 1
                if i - last > merge_window or last_side * side == -1:
                    if last - first > 0:
                        write_interval_to_filter(
                            chromosome,
                            first,
                            last,
                        )
                    first = i
                last = i
                last_side = side
            else:
                if window_depth < mu:
                    side = -1
                    p = norm_dist.cdf(window_depth)

                    if p >= p_threshold:
                        keep_threshold[0] = window_depth
                    else:
                        filter_threshold[0] = window_depth
                        if i - last > merge_window or last_side * side == -1:
                            if last - first > 0:
                                write_interval_to_filter(
                                    chromosome,
                                    first,
                                    last,
                                )
                            first = i
                        last = i
                        last_side = side

                elif window_depth > mu:
                    side = 1
                    p = 1. - norm_dist.cdf(window_depth)

                    if p >= p_threshold:
                        keep_threshold[1] = window_depth
                    else:
                        filter_threshold[1] = window_depth
                        if i - last > merge_window or last_side * side == -1:
                            if last - first > 0:
                                write_interval_to_filter(
                                    chromosome,
                                    first,
                                    last,
                                )
                            first = i
                        last = i
                        last_side = side
    if last - first > 0:
        write_interval_to_filter(
            chromosome,
            first,
            last,
        )


def filter_regions_by_depth(depths, mu, sigma,
                            filtered_regions_output, p_threshold=0.0001,
                            merge_window=1000): # removed unused chrom_sizes argument
    '''
    Filter positions by depth observing a normal distribution.  See
    process_chromosome_values for additional details.
    '''
    with open(filtered_regions_output, 'w') as OUT:

        for chromosome in sorted(depths.keys()):

            process_chromosome_values(
                chromosome, depths[chromosome], mu, sigma, OUT, \
                p_threshold, merge_window)


def filter_regions_by_depth_bedgraph(bedgraph_file, chrom_sizes, mu,
                                     sigma, filtered_regions_output,
                                     ploidy=2, cnv_bedgraph_file=None,
                                     p_threshold=0.0001, merge_window=1000):
    '''
    Pass depths read from a bedGraph file to filter_regions_by_depth.
    '''
    depths = dict()

    if cnv_bedgraph_file is not None:
        cnv_regions = read_cnv_bedgraph(cnv_bedgraph_file)
    else:
        cnv_regions = dict()

    with open(bedgraph_file) as f:
        for line in f:

            chromosome, start, end, coverage = line.strip().split()
            start = int(start) + 1  # Convert from zero-based
            end = int(end)
            coverage = float(coverage)

            if chromosome not in depths:
                depths[chromosome] = numpy.zeros(chrom_sizes[chromosome], \
                    dtype=numpy.int32)

            for position in range(start, end + 1):
                if (chromosome, position) in cnv_regions:
                    depths[chromosome][position - 1] = int(coverage / \
                        cnv_regions[(chromosome, position)])
                else:
                    depths[chromosome][position - 1] = int(coverage / ploidy)

    filter_regions_by_depth(depths, mu, sigma, \
        filtered_regions_output, p_threshold, merge_window)
    
    
def filter_regions_by_depth_mpileup(mpileup_file, chrom_sizes, mu,
                                     sigma, filtered_regions_output,
                                     ploidy, cnv_regions, p_threshold=0.0001,
                                     merge_window=1000):
    '''
    Pass depths read from a mpileup TXT file to filter_regions_by_depth.
    '''
    depths = dict()

    with open(mpileup_file) as f:
        for line in f:

            line_split = line.strip().split()

            chromosome, position, reference_base, coverage = line_split[:4]
            position = int(position)
            coverage = float(coverage)

            if chromosome not in depths:
                depths[chromosome] = numpy.zeros(chrom_sizes[chromosome], \
                    dtype=numpy.int32)

            if int(coverage) > 0:
                bases = line_split[4]
            else:
                bases = ''

            i = 0
            while i < len(bases):
                if bases[i] == '^':
                    i += 1
                elif bases[i] == '*':
                    coverage += 1
                i += 1

            if coverage > 0:
                if (chromosome, position) in cnv_regions:
                    depths[chromosome][position - 1] = int(coverage / \
                        cnv_regions[(chromosome, position)])
                else:
                    depths[chromosome][position - 1] = int(coverage / ploidy)

    filter_regions_by_depth(depths, mu, sigma,
        filtered_regions_output, p_threshold, merge_window)
