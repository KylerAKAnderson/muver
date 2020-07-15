from collections import defaultdict, OrderedDict
import copy
import math
import re
from scipy.stats import binom, chi2
import sys

from fitting import logistic


def get_repeat_adjustment_value(unit_length, repeat_sequence_length,
                                 event_type, fits):
    '''
    Gives an expected frequency of observing an indel in a repeat for a given
    repeat unit length and repeat tract length. Expects parameters from log-
    transformed rates fit to a logistic function.
    '''
    fit = fits[event_type][unit_length]

    return 10 ** min(0, logistic(
        repeat_sequence_length,
        fit['x0'],
        fit['L'],
        fit['M'],
        fit['k'],
    ))


def genotype_to_allele_counts(genotype, alleles):
    '''
    Finds the occurrences of each allele in a given genotype. Returns counts in
    a dict.
    '''
    allele_counts = dict()

    for allele in alleles:
        allele_counts[allele] = genotype.count(allele)

    return allele_counts


def get_possible_genotypes(ploidy, alleles):
    '''
    For given ploidy and alleles, return a list of all possible genotypes.
    '''
    possible_genotypes = []

    genotypes = [[]]
    for i in range(ploidy):
        temp_genotypes = []
        for allele in alleles:
            for genotype in genotypes:
                temp_genotype = copy.deepcopy(genotype)
                temp_genotype.append(allele)
                temp_genotypes.append(temp_genotype)
        genotypes = copy.deepcopy(temp_genotypes)

    for genotype in genotypes:
        _sorted = tuple(sorted(genotype, key=lambda x: alleles.index(x)))
        if _sorted not in possible_genotypes:
            possible_genotypes.append(_sorted)

    return possible_genotypes


def loh_comparison(loh_1, loh_2):
    '''
    Compare two LOH values. Considers ternary logic.
    '''
    lohs = [loh_1, loh_2]
    if None in lohs:
        return None
    elif True in lohs and False in lohs:
        return None
    elif lohs == [True, True]:
        return True
    elif lohs == [False, False]:
        return False
    else:
        raise ValueError('LOH comparison failed.')


def resolve_loh(loh):
    '''
    Return an LOH value in integer notation.
    '''
    if loh is None:
        return -1
    elif loh is True:
        return 1
    elif loh is False:
        return 0


def get_mutation_name(position, reference_allele, start_allele, end_allele,
                      ambiguous=False):
    '''
    Get the name for a mutation given the start allele and end allele.
    '''
    # Allele gain (CNV)
    if end_allele and not start_allele:
        if ambiguous:
            return 'g.{}gain{}'.format(str(position), '*')
        else:
            return 'g.{}gain{}'.format(str(position), end_allele)
    # Allele loss (CNV)
    elif start_allele and not end_allele:
        if ambiguous:
            return 'g.{}loss{}'.format(str(position), '*')
        else:
            return 'g.{}loss{}'.format(str(position), start_allele)
    # Allele conversion
    else:
        start = position
        r = reference_allele
        a_1 = start_allele
        a_2 = end_allele

        if r and a_1 and a_2:
            while r[-1] == a_1[-1] and r[-1] == a_2[-1]:
                r = r[:-1]
                a_1 = a_1[:-1]
                a_2 = a_2[:-1]
                if not r or not a_1 or not a_2:
                    break

        if r and a_1 and a_2:
            while r[0] == a_1[0] and r[0] == a_2[0]:
                r = r[1:]
                a_1 = a_1[1:]
                a_2 = a_2[1:]
                start += 1
                if not r or not a_1 or not a_2:
                    break

        end = start + max(len(r) - 1, 0)
        if r == a_1:
            if a_1 == '':
                event = 'ins' + a_2
                start -= 1
            elif a_2 == '':
                event = 'del' + a_1
            else:
                event = a_1 + '>' + a_2
        else:
            event = a_1 + '>' + a_2
            end += 1
            start -= 1

        if end > start:
            position = '{}_{}'.format(str(start), str(end))
        else:
            position = str(start)

        return 'g.{}{}'.format(position, event)


class Variant(object):
    '''
    Class for MuVer variants.
    '''
    def __init__(self, chromosome, position, alleles, samples, ref_allele,
            control_sample, strand_allele_counts):

        self.chromosome = chromosome
        self.position = int(position)
        self.alleles = alleles
        self.samples = samples
        self.ref_allele = ref_allele
        self.control_sample = control_sample
        self.strand_allele_counts = strand_allele_counts

        self.get_sample_total_counts()

    def get_sample_total_counts(self):
        '''
        For each sample, go through strand allele counts and find the total
        counts. Assign totals to self.sample_total_counts (dict).
        '''
        self.sample_total_counts = dict()
        sac = self.strand_allele_counts

        for sample in self.samples:
            total_counts = 0
            for allele in self.alleles:
                for count in sac[sample][allele].values():
                    total_counts += count
            self.sample_total_counts[sample] = total_counts

    def check_if_excluded(self, excluded_regions):
        '''
        Compare position to excluded regions. Assign overlap to excluded_flag.
        '''
        pos = (self.chromosome, self.position)
        if excluded_regions:
            self.excluded_flag = pos in excluded_regions
        else:
            self.excluded_flag = False

    def check_depth_threshold(self, depth_threshold):
        '''
        For each sample, see if read depth is greater than the threshold.
        '''
        self.depth_flags = dict()
        for sample in self.samples:
            self.depth_flags[sample] = \
                self.sample_total_counts[sample] >= depth_threshold

    def check_filtered_sites(self, filtered_sites):
        '''
        For each sample, see if the position overlaps a filtered site.
        '''
        self.filtered_sites_flags = dict()
        for sample in self.samples:
            self.filtered_sites_flags[sample] = \
                (self.chromosome, self.position) in filtered_sites[sample]

    def check_allele_coverage(self):
        '''
        For each sample, flag if more than one allele/strand pair has coverage.
        '''
        self.allele_coverage_flags = dict()
        sac = self.strand_allele_counts

        for sample in self.samples:

            count = 0
            for allele in self.alleles:
                for depth in sac[sample][allele].values():
                    if depth > 0:
                        count += 1

            self.allele_coverage_flags[sample] = count > 1

    def binomial_test(self):
        '''
        For each sample, perform binomial tests for each strand allele count
        relative to control.
        '''
        self.binomial_p_values = dict()
        sac = self.strand_allele_counts
        control_sum = self.sample_total_counts[self.control_sample]

        for sample in [s for s in self.samples if s != self.control_sample]:
            sample_sum = self.sample_total_counts[sample]
            self.binomial_p_values[sample] = dict()

            for allele in self.alleles:
                self.binomial_p_values[sample][allele] = dict()
                for strand, sample_value in sac[sample][allele].items():

                    if (
                        sample_value > 0 and
                        self.allele_coverage_flags[sample] and
                        self.allele_coverage_flags[self.control_sample]
                    ):
                        control_value = max(
                            sac[self.control_sample][allele][strand],
                            1,
                        )

                        self.binomial_p_values[sample][allele][strand] = \
                            binom.cdf(
                                sample_sum - int(sample_value),
                                sample_sum,
                                1.0 - (float(control_value) / control_sum),
                            )
                    else:
                        self.binomial_p_values[sample][allele][strand] = \
                            1.0

    def chisquare_test(self):
        '''
        For each sample, perform chi-squared test relative to control.
        '''
        self.chisquare_p_values = dict()
        sac = self.strand_allele_counts
        control_sum = self.sample_total_counts[self.control_sample]

        for sample in [s for s in self.samples if s != self.control_sample]:
            if (
                self.allele_coverage_flags[sample] and
                self.allele_coverage_flags[self.control_sample]
            ):
                chi_sum = 0.0
                df = 0
                sample_sum = self.sample_total_counts[sample]

                for allele in self.alleles:
                    for strand, sample_value in sac[sample][allele].items():
                        control_value = sac[self.control_sample][allele][strand]

                        if sample_value > 0 or control_value > 0:
                            c = float(control_value)
                            s = float(sample_value)
                            c_sum = float(control_sum)
                            s_sum = float(sample_sum)

                            chi_sum += ((s * math.sqrt(c_sum / s_sum) -
                                c * math.sqrt(s_sum / c_sum)) ** 2) / \
                                (s + c)
                            df += 1

                self.chisquare_p_values[sample] = 1.0 - chi2.cdf(chi_sum, df)
            else:
                self.chisquare_p_values[sample] = 1.0

    def check_if_significant(self, p_threshold):
        '''
        For each sample, consider binomial and chi-squared tests relative to
        the p-value to determine if the genotype is significantly different
        than the control.
        '''
        self.sample_significance_flags = dict()
        self.sample_composite_p_values = dict()

        binom = self.binomial_p_values
        chi = self.chisquare_p_values

        log_p = -math.log10(p_threshold)

        for sample in [s for s in self.samples if s != self.control_sample]:

            significance_flag = False
            _chi = max(chi[sample], sys.float_info.min)
            composite_p_values = []

            for allele in self.alleles:

                _binom_forward = max(binom[sample][allele]['forward'],
                    sys.float_info.min)
                _binom_reverse = max(binom[sample][allele]['reverse'],
                    sys.float_info.min)

                comp = math.sqrt(
                    (-math.log10(_binom_forward)) ** 2 +
                    (-math.log10(_binom_reverse)) ** 2 +
                    (-math.log10(_chi)) ** 2
                )
                if (
                    comp >= log_p and
                    _binom_forward < 0.1 and
                    _binom_reverse < 0.1 and
                    _chi < 0.1
                ):
                    significance_flag = True

                composite_p_values.append(comp)

            self.sample_composite_p_values[sample] = max(composite_p_values)
            self.sample_significance_flags[sample] = significance_flag

    def assign_ploidy(self):
        '''
        For each sample, assign ploidy.
        '''
        self.sample_ploidy = dict()
        position = (self.chromosome, self.position)

        for sample in self.samples:
            self.sample_ploidy[sample] = sample.ploidy

            if position in sample.cnv_regions:
                self.sample_ploidy[sample] = \
                    sample.cnv_regions[position]

    def intersect_with_repeats(self, repeats):
        '''
        Check if the position of the variant intersects with a repeat.
        '''
        try:
            self.intersected_repeats = repeats[self.chromosome][self.position]
        except:
            self.intersected_repeats = None
        self.intersect_repeat_flag = bool(self.intersected_repeats)

    def find_repeat_expanded_alleles(self):
        '''
        If the variant is left-aligned with a repeat, append the rest of the
        repeat sequence to the allele, and assign to repeat_expanded_alleles.
        '''
        self.intersected_repeat_unit = None
        self.intersected_repeat_added_sequence = None
        self.repeat_expanded_ref_allele = self.ref_allele

        self.repeat_expanded_alleles = dict()
        for allele in self.alleles:
            self.repeat_expanded_alleles[allele] = allele

        if self.intersected_repeats:

            left_aligned_repeats = [r for r in self.intersected_repeats
                if r['start'] == self.position]

            if left_aligned_repeats:

                repeat = sorted(
                    left_aligned_repeats, key=lambda x: len(x['unit']))[0]

                expand = False
                for allele in [a for a in self.alleles
                        if a != self.ref_allele]:
                    if len(allele) != self.ref_allele:

                        repeat_insertion = \
                            re.match(
                                allele + '(' + repeat['unit'] + ')+',
                                self.ref_allele,
                            )
                        repeat_deletion = \
                            re.match(
                                self.ref_allele + '(' + repeat['unit'] + ')+',
                                allele,
                            )

                        if repeat_insertion or repeat_deletion:
                            expand = True
                            break

                if expand:

                    seq = repeat['sequence']
                    unit = repeat['unit']

                    units_in_sequence = len(seq.split(unit)) - 1
                    units_in_ref_allele = len(self.ref_allele.split(unit)) - 1

                    to_add = ''
                    for i in range(units_in_sequence - units_in_ref_allele):
                        to_add += unit

                    self.intersected_repeat_unit = repeat['unit']
                    self.intersected_repeat_added_sequence = to_add
                    self.repeat_expanded_ref_allele = self.ref_allele + to_add
                    for allele in self.alleles:
                        self.repeat_expanded_alleles[allele] = allele + to_add

    def call_genotypes_and_subclonal_alleles(self):
        '''
        Based on observed and expected allelic frequencies, call clonal and
        subclonal genotypes.
        '''
        self.expected_allele_frequencies = dict()
        self.sample_genotypes = dict()
        self.sample_genotype_min_log_ratio_sum = dict()
        self.sample_subclonal_alleles = dict()
        self.sample_subclonals = dict()

        for sample in [self.control_sample] + \
                [s for s in self.samples if s != self.control_sample]:
            if (self.depth_flags[sample] and not self.excluded_flag
                    and self.sample_ploidy[sample] != 0 and self.sample_ploidy[sample] < 11):

                ploidy = self.sample_ploidy[sample]
                allele_counts = self.strand_allele_counts[sample]
                possible_genotypes = get_possible_genotypes(
                    ploidy, self.alleles)

                observed_allele_frequencies = dict()
                total_sample_counts = 0

                for allele, counts in allele_counts.items():
                    observed_allele_frequencies[allele] = 0
                    for value in counts.values():
                        observed_allele_frequencies[allele] += max(value, 1)
                        total_sample_counts += value
                for allele, frequency in observed_allele_frequencies.items():
                    observed_allele_frequencies[allele] = \
                        float(frequency) / total_sample_counts

                self.expected_allele_frequencies[sample] = OrderedDict()
                eaf = self.expected_allele_frequencies[sample]

                #  No subclonal
                for genotype in possible_genotypes:
                    eaf[(genotype, (None, None, None))] = dict()

                    for allele in self.alleles:
                        allele_count = genotype.count(allele)
                        eaf[(genotype, (None, None, None))][allele] = \
                            float(allele_count) / len(genotype)

                # With subclonals
                for subclonal_allele in self.alleles:

                    for genotype in possible_genotypes:

                        subclonal_genotypes = []

                        for i, allele in enumerate(genotype):

                            if allele != subclonal_allele:

                                subclonal_genotype = list(genotype)
                                subclonal_genotype[i] = subclonal_allele

                                _sorted = tuple(sorted(
                                    subclonal_genotype,
                                    key=lambda x: self.alleles.index(x),
                                ))
                                if _sorted not in subclonal_genotypes:
                                    subclonal_genotypes.append(_sorted)

                        for subclonal_genotype in subclonal_genotypes:
                            for subclonal_frequency in [0.500, 0.250, 0.125]:

                                subclonal = (
                                    subclonal_genotype,
                                    subclonal_allele,
                                    subclonal_frequency,
                                )

                                eaf[(genotype, subclonal)] = dict()

                                for allele in self.alleles:

                                    eaf[(genotype, subclonal)][allele] = \
                                        (1 - subclonal_frequency) * (float(genotype.count(allele)) / len(genotype)) + \
                                        subclonal_frequency * (float(subclonal_genotype.count(allele)) / len(subclonal_genotype))

                if self.intersected_repeat_unit:

                    fits = sample.repeat_indel_fits_dict

                    for frequencies in eaf.values():

                        frequency_adjustment_values = dict()

                        for allele in frequencies.keys():
                            frequency_adjustment_values[allele] = 0.0

                        for allele, frequency in frequencies.items():
                            re_allele = self.repeat_expanded_alleles[allele]

                            plus_one = re_allele + self.intersected_repeat_unit
                            minus_one = re_allele.replace(
                                self.intersected_repeat_unit, '', 1)

                            unit_length = len(self.intersected_repeat_unit)
                            repeat_sequence_length = unit_length * \
                                re_allele.count(self.intersected_repeat_unit)

                            plus_one_adj = max(get_repeat_adjustment_value(
                                unit_length, repeat_sequence_length, 'insertion', fits), 0) * frequency
                            minus_one_adj = max(get_repeat_adjustment_value(
                                unit_length, repeat_sequence_length, 'deletion', fits), 0) * frequency

                            if plus_one in self.repeat_expanded_alleles.values():

                                for al, re in self.repeat_expanded_alleles.items():
                                    if re == plus_one:
                                        plus_one_allele = al

                                frequency_adjustment_values[plus_one_allele] += plus_one_adj
                                frequency_adjustment_values[allele] -= plus_one_adj

                            if minus_one != allele and minus_one in self.repeat_expanded_alleles.values():
                                for al, re in self.repeat_expanded_alleles.items():
                                    if re == minus_one:
                                        minus_one_allele = al

                                frequency_adjustment_values[minus_one_allele] += minus_one_adj
                                frequency_adjustment_values[allele] -= minus_one_adj

                        for allele, adj in frequency_adjustment_values.items():
                            frequencies[allele] += adj

                for frequencies in eaf.values():
                    for allele, frequency in frequencies.items():
                        frequencies[allele] = max(frequency, 2.0 / total_sample_counts)

                min_log_ratio_sum = float('inf')
                max_shared_count = 0

                for key, ef in eaf.items():
                    genotype, subclonal = key
                    log_ratio_sum = 0
                    shared_count = 0

                    for allele, frequency in ef.items():
                        log_ratio_sum += abs(math.log(observed_allele_frequencies[allele] / frequency))

                    for allele in genotype:
                        if sample == self.control_sample and allele == self.ref_allele:
                            shared_count += 1
                        if sample != self.control_sample and self.sample_genotypes[self.control_sample]:
                            if allele in self.sample_genotypes[self.control_sample]:
                                shared_count += 1

                    # TODO: clean this up
                    if log_ratio_sum < (min_log_ratio_sum - 0.000001) or \
                            (abs(log_ratio_sum - min_log_ratio_sum) < 0.000001 and shared_count > max_shared_count):

                        min_log_ratio_sum = log_ratio_sum
                        max_shared_count = shared_count

                        self.sample_genotypes[sample] = genotype
                        self.sample_genotype_min_log_ratio_sum[sample] = min_log_ratio_sum
                        if subclonal == (None, None, None):
                            self.sample_subclonal_alleles[sample] = None
                            self.sample_subclonals[sample] = {
                                'genotype': None,
                                'frequency': None,
                            }
                        else:
                            self.sample_subclonal_alleles[sample] = subclonal
                            self.sample_subclonals[sample] = {
                                'genotype': subclonal[0],
                                'frequency': subclonal[2],
                            }

            else:
                self.sample_genotypes[sample] = None
                self.sample_genotype_min_log_ratio_sum[sample] = None
                self.sample_subclonal_alleles[sample] = None

                self.sample_subclonals[sample] = {
                    'genotype': None,
                    'frequency': None,
                }

    def subclonal_strand_bias_log_normal_test(self, samples):
        '''
        For the given subclonal call, check the strand balance given a log-
        normal distribution.
        '''
        self.sample_subclonal_bias_log_normal = dict()

        for sample in samples:

            subclonal = self.sample_subclonal_alleles[sample]
            genotype = self.sample_genotypes[sample]
            allele_counts = self.strand_allele_counts[sample]

            if subclonal:

                subclonal_allele = subclonal[1]
                f = max(allele_counts[subclonal_allele]['forward'], 1)
                r = max(allele_counts[subclonal_allele]['reverse'], 1)
                strand_bias_stdev = sample.strand_bias_std

                if f == 0 or r == 0:
                    log_normal_p_value = 0.0
                else:
                    log_ratio = -abs(math.log(float(f) / r))
                    log_normal_p_value = 2 * 0.5 * \
                        (1 + math.erf(log_ratio / (math.sqrt(2) * strand_bias_stdev)))

            else:
                log_normal_p_value = None

            self.sample_subclonal_bias_log_normal[sample] = \
                log_normal_p_value

    def subclonal_binomial_test(self, p_threshold):
        '''
        Perform binomial test for the subclonal allele comparing against the
        frequency expected given the called genotype.
        '''
        self.sample_subclonal_binomial = dict()

        for sample in self.samples:
            self.sample_subclonal_binomial[sample] = None

            subclonal = self.sample_subclonal_alleles[sample]
            genotype = self.sample_genotypes[sample]
            allele_counts = self.strand_allele_counts[sample]

            if genotype and subclonal:
                eaf = self.expected_allele_frequencies[sample]
                subclonal_allele = subclonal[1]

                f = max(allele_counts[subclonal_allele]['forward'], 1)
                r = max(allele_counts[subclonal_allele]['reverse'], 1)

                sample_sum = 0
                for counts in allele_counts.values():
                    sample_sum += sum(counts.values())

                binomial_p_value = binom.cdf(
                    sample_sum - f - r,
                    sample_sum,
                    1.0 - eaf[genotype, (None, None, None)][subclonal_allele],
                )

                #if binomial_p_value > p_threshold:
                #    self.sample_subclonal_alleles[sample] = None
                #    self.sample_subclonals[sample] = {
                #        'genotype': None,
                #        'frequency': None,
                #    }
            else:
                binomial_p_value = None

            self.sample_subclonal_binomial[sample] = binomial_p_value

    def subclonal_strand_bias_binomial_test(self):
        '''
        Perform binomial test for strand bias in the subclonal allele.
        '''
        self.sample_subclonal_bias_binomial = dict()

        for sample in self.samples:

            subclonal = self.sample_subclonal_alleles[sample]
            genotype = self.sample_genotypes[sample]
            allele_counts = self.strand_allele_counts[sample]

            if subclonal:

                subclonal_allele = subclonal[1]
                f = max(allele_counts[subclonal_allele]['forward'], 1)
                r = max(allele_counts[subclonal_allele]['reverse'], 1)

                if f == 0 or r == 0:
                    binomial_p_value = 0.0
                else:
                    binomial_p_value = binom.cdf(
                        min(f, r),
                        sum((f, r)),
                        0.5,
                    )

            else:
                binomial_p_value = None

            self.sample_subclonal_bias_binomial[sample] = binomial_p_value

    def _get_mutation_name(self, start_allele, end_allele, ambiguous=False):
        return get_mutation_name(self.position, self.ref_allele, start_allele,
                                 end_allele, ambiguous=ambiguous)

    def get_all_possible_mutation_transitions(self):
        '''
        Given the observed alleles, create a list of all possible mutations.
        '''
        mutations = []

        for allele_1 in self.alleles:

            #  Gain of an allele
            mutations.append({
                'name': self._get_mutation_name(
                    None, allele_1, ambiguous=False),
                'start_allele': None,
                'end_allele': allele_1,
                'transitions': {allele_1: 1},
                'type': 'gain',
            })
            #  Loss of an allele
            mutations.append({
                'name': self._get_mutation_name(
                    allele_1, None, ambiguous=False),
                'start_allele': allele_1,
                'end_allele': None,
                'transitions': {allele_1: -1},
                'type': 'loss',
            })

            #  Allele 'conversion': one allele to another
            for allele_2 in [a for a in self.alleles if a != allele_1]:
                mutations.append({
                    'name': self._get_mutation_name(
                        allele_1, allele_2, ambiguous=False),
                    'transitions': {allele_1: -1, allele_2: 1},
                    'start_allele': allele_1,
                    'end_allele': allele_2,
                    'type': 'conversion',
                })

        return mutations

    def get_mutation_paths(self, sample_genotype, control_genotype, mutations):
        '''
        For the given sample and control genotypes, describe all valid mutation
        paths.
        '''
        start_allele_counts = \
            genotype_to_allele_counts(control_genotype, self.alleles)
        end_allele_counts = \
            genotype_to_allele_counts(sample_genotype, self.alleles)

        paths = [{
            'endpoint': start_allele_counts,
            'mutations': [],
            'loh': []
        }]

        while end_allele_counts not in [p['endpoint'] for p in paths]:

            temp_paths = []

            for path in paths:
                for mutation in mutations:

                    temp_path = copy.deepcopy(path)
                    valid = True

                    #  Check for gain of allele not present
                    if mutation['type'] == 'gain':
                        gained_allele = list(mutation['transitions'].keys())[0]
                        if path['endpoint'][gained_allele] < 1:
                            valid = False # it looks as if all the information from here to 'if valid' is meaningless under this condition, so i think we can just continue

                    #  Determine LOH flag
                    loh_flag = False
                    if sum(mutation['transitions'].values()) == 0:
                        for allele, value in mutation['transitions'].items():
                            if all([
                                value == 1,
                                temp_path['endpoint'][allele] > 0,
                            ]):
                                loh_flag = True

                    #  Apply mutation to endpoint
                    for allele, value in mutation['transitions'].items():
                        temp_path['endpoint'][allele] += value
                    temp_path['mutations'].append(mutation)
                    temp_path['loh'].append(loh_flag)

                    #  Check for negative allele values
                    for value in temp_path['endpoint'].values():
                        if value < 0:
                            valid = False

                    if valid:
                        temp_paths.append(temp_path)

            paths = copy.deepcopy(temp_paths)

        return [p for p in paths if p['endpoint'] == end_allele_counts]

    @staticmethod
    def analyze_paths(paths):
        '''
        Analyze paths to find shared mutations. Analysis is anchored on the
        first path. If a mutation in the first path is not shared, it is
        returned in the orphan list for subsequent analysis.
        '''
        shared_mutations = []
        orphan_mutations = []

        for mutation, loh in zip(paths[0]['mutations'], paths[0]['loh']):
            shared = True
            for path in paths[1:]:
                if mutation not in path['mutations']:
                    shared = False
            if shared:
                for i, path in enumerate(copy.deepcopy(paths)):
                    if i > 0:
                        index = next(
                            (j for j, m in enumerate(path['mutations'])
                                if m == mutation),
                            None,
                        )
                        loh = loh_comparison(loh, path['loh'][index])
                        paths[i]['mutations'].pop(index)
                        paths[i]['loh'].pop(index)
                shared_mutations.append((mutation, resolve_loh(loh)))
            else:
                orphan_mutations.append((mutation, resolve_loh(loh)))

        return shared_mutations, orphan_mutations

    @staticmethod
    def check_gains(mutation, **kwargs):
        return mutation['type'] == 'gain'

    @staticmethod
    def check_losses(mutation, **kwargs):
        return mutation['type'] == 'loss'

    @staticmethod
    def check_conversions(mutation, **kwargs):
        start_allele = kwargs['start_allele']
        return all([
            mutation['type'] == 'conversion',
            mutation['start_allele'] == start_allele,
        ])

    def resolve_orphan_mutations(self, orphans, paths, comparison_function):
        shared_mutations = []
        for mutation, loh in orphans:

            start_allele = mutation['start_allele']
            start_shared = True
            for path in paths[1:]:
                if not list(j for j, m in enumerate(path['mutations']) if comparison_function(m, start_allele=start_allele)):
                    start_shared = False

            if start_shared:

                similar_mutations = [mutation]
                mutation_loh = {frozenset(mutation): loh}

                for i, path in enumerate(copy.deepcopy(paths)):
                    if i > 0:

                        index = next(j for j, m in enumerate(path['mutations']) if comparison_function(m, start_allele=start_allele))

                        similar_mutation = paths[i]['mutations'].pop(index)
                        similar_loh = loh_comparison(loh, paths[i]['loh'].pop(index))

                        if frozenset(similar_mutation) in mutation_loh:
                            _loh = mutation_loh[frozenset(similar_mutation)]
                            mutation_loh[frozenset(similar_mutation)] = \
                                loh_comparison(similar_loh, _loh)
                        else:
                            mutation_loh[frozenset(similar_mutation)] = \
                                similar_loh

                        if similar_mutation not in similar_mutations:
                            similar_mutations.append(similar_mutation)

                similar_mutations.sort(key=lambda x: x['name'])
                mutation_names = list(
                    self._get_mutation_name(m['start_allele'], m['end_allele'], ambiguous=True) for m in similar_mutations)
                mutation_lohs = list(
                    str(resolve_loh(mutation_loh[frozenset(m)])) for m in similar_mutations)

                shared_mutations.append((
                    {'name': '|'.join(mutation_names), 'type': 'conversion'},
                    '|'.join(mutation_lohs),
                ))

        return shared_mutations

    def call_mutations(self):
        '''
        For each sample, call mutations.
        '''
        self.sample_called_mutations = dict()
        self.sample_called_loh = dict()

        possible_mutations = self.get_all_possible_mutation_transitions()

        for sample in [s for s in self.samples if s != self.control_sample]:

            self.sample_called_mutations[sample] = []
            self.sample_called_loh[sample] = []

            shared_mutations = self.sample_called_mutations[sample]
            shared_loh_flags = self.sample_called_loh[sample]
            sample_genotype = self.sample_genotypes[sample]
            control_genotype = self.sample_genotypes[self.control_sample]

            if sample_genotype and control_genotype:
                if sample_genotype != control_genotype:

                    paths = self.get_mutation_paths(
                        sample_genotype, control_genotype, possible_mutations)

                    _shared, orphans = self.analyze_paths(paths)
                    if _shared:
                        _mutations, _loh_flags = zip(*_shared)
                        shared_mutations.extend(_mutations)
                        shared_loh_flags.extend(_loh_flags)

                    orphan_gains = \
                        (m for m in orphans if m[0]['type'] == 'gain')
                    orphan_losses = \
                        (m for m in orphans if m[0]['type'] == 'loss')
                    orphan_conversions = \
                        (m for m in orphans if m[0]['type'] == 'conversion')

                    for _list, _function in (
                        (orphan_gains, self.check_gains),
                        (orphan_losses, self.check_losses),
                        (orphan_conversions, self.check_conversions),
                    ):
                        _shared = self.resolve_orphan_mutations(
                            _list,
                            paths,
                            _function,
                        )
                        if _shared:
                            _mutations, _loh_flags = zip(*_shared)
                            shared_mutations.extend(_mutations)
                            shared_loh_flags.extend(_loh_flags)

                    self.sample_called_mutations[sample] = shared_mutations
                    self.sample_called_loh[sample] = shared_loh_flags

    def set_report_mutations_flag(self, p_threshold):
        '''
        Check object attributes to see if a mutation should be reported.

        To be reported, the following must be true:

        - Not within excluded regions.
        - Control and sample not in a filtered region.
        - Control and sample above depth threshold.
        - The control subclonal strand bias log-normal p-value is not less than
            threshold.
        - More than one allele/strand pair displays read coverage.
        - Passes the significance threshold.
        '''
        self.report_mutations = dict()

        if self.sample_subclonals[self.control_sample]['genotype']:
            p = self.sample_subclonal_bias_log_normal[self.control_sample]
            control_subclonal_check = p >= p_threshold
        else:
            control_subclonal_check = True

        for sample in self.sample_called_mutations:
            self.report_mutations[sample] = all([
                not self.excluded_flag,
                not self.filtered_sites_flags[sample],
                not self.filtered_sites_flags[self.control_sample],
                self.depth_flags[sample],
                self.depth_flags[self.control_sample],
                control_subclonal_check,
                self.allele_coverage_flags[sample],
                self.sample_significance_flags[sample],
            ])

    def check_subclonal_validity(self, p_threshold):
        '''
        Check object attributes to see if a subclonal call is valid.

        To be valid, the following must be true:

        - Not within excluded regions.
        - Not in a filtered region.
        - More than one allele/strand pair displays read coverage.
        - The subclonal strand bias log-normal p-value is not less than
            threshold.
        - The subclonal depth binomial test is less than threshold.
        '''
        self.sample_subclonal_valid_flag = dict()

        for sample in self.sample_subclonals:
            self.sample_subclonal_valid_flag[sample] = all([
                not self.excluded_flag,
                not self.filtered_sites_flags[sample],
                self.allele_coverage_flags[sample],
                (self.sample_subclonal_bias_log_normal[sample] is not None and \
                self.sample_subclonal_bias_log_normal[sample] >= p_threshold),
                (self.sample_subclonal_binomial[sample] is not None and \
                self.sample_subclonal_binomial[sample] < p_threshold),
            ])
