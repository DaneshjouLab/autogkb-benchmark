"""
PGxMine variant normalization.

Port of PGxMine's normalizeMutation() function from:
https://github.com/jakelever/pgxmine/blob/main/utils/__init__.py

This module implements PGxMine's comprehensive normalization strategy with
157+ regex patterns for variant forms including:
- Star alleles (CYP2D6*4, NUDT15*3)
- rsIDs (rs9923231)
- Protein variants (p.T790M, THR790MET)
- DNA/cDNA variants (c.93G>A, g.93delG)
"""

import re

# Amino acid mappings: 3-letter codes, full names, and single-letter codes
AMINO_ACID_INFO = [
    ('ALA', 'A'), ('ARG', 'R'), ('ASN', 'N'), ('ASP', 'D'), ('CYS', 'C'),
    ('GLU', 'E'), ('GLN', 'Q'), ('GLY', 'G'), ('HIS', 'H'), ('ILE', 'I'),
    ('LEU', 'L'), ('LYS', 'K'), ('MET', 'M'), ('PHE', 'F'), ('PRO', 'P'),
    ('SER', 'S'), ('THR', 'T'), ('TRP', 'W'), ('TYR', 'Y'), ('VAL', 'V'),
    ('ALANINE', 'A'), ('CYSTEINE', 'C'), ('ASPARTICACID', 'D'),
    ('GLUTAMICACID', 'E'), ('PHENYLALANINE', 'F'), ('GLYCINE', 'G'),
    ('HISTIDINE', 'H'), ('ISOLEUCINE', 'I'), ('LYSINE', 'K'),
    ('LEUCINE', 'L'), ('METHIONINE', 'M'), ('ASPARAGINE', 'N'),
    ('PROLINE', 'P'), ('GLUTAMINE', 'Q'), ('ARGININE', 'R'),
    ('SERINE', 'S'), ('THREONINE', 'T'), ('VALINE', 'V'),
    ('TRYPTOPHAN', 'W'), ('TYROSINE', 'Y'), ('STOP', 'X'), ('TER', 'X')
]

AMINO_ACID_MAP = {big: small for big, small in AMINO_ACID_INFO}
# Add single letter mappings
for letter in 'ABCDEFGHIKLMNPQRSTVWYZX':
    AMINO_ACID_MAP[letter] = letter
AMINO_ACID_MAP['*'] = '*'


def normalize_mutation(mention: str) -> str | None:
    """Normalize a variant mention using PGxMine's 157 regex patterns.

    Args:
        mention: Raw variant text (e.g., "THR790MET", "93G>A", "*4")

    Returns:
        Normalized variant (e.g., "p.T790M", "c.93G>A", "*4") or None if no match

    Examples:
        >>> normalize_mutation("THR790MET")
        'p.T790M'
        >>> normalize_mutation("c.93G>A")
        'c.93G>A'
        >>> normalize_mutation("*4")
        '*4'
    """
    # Star alleles and rsIDs: just remove spaces
    if mention.strip().startswith('*'):
        return mention.replace(' ', '')
    elif mention.startswith('rs'):
        return mention.replace(' ', '')

    # Pattern examples with their normalized output formats
    # Each tuple is (output_format, input_pattern)
    examples = [
        # Protein variants - simple notation
        ('p.T790M', 'p.T790M'),
        ('p.T790M', 'p.(T790M)'),
        ('p.T790M', '790T>M'),
        ('p.T790M', '790T->M'),
        ('p.T790M', '790T-->M'),
        ('p.T790M', 'T790->M'),
        ('p.T790M', 'T790-->M'),

        # Protein variants - three-letter codes
        ('p.T790M', 'THR790MET'),
        ('p.T790M', 'THR790/MET'),
        ('p.T790M', 'THR790 to MET'),
        ('p.T790M', 'THR-790 to MET'),
        ('p.T790M', 'THR790-to-MET'),
        ('p.T790M', 'THR790->MET'),
        ('p.T790M', 'THR790-->MET'),
        ('p.T790M', 'THR790-MET'),
        ('p.T790M', 'THR790----MET'),
        ('p.T790M', '790THR----MET'),
        ('p.T790M', 'THR-790-MET'),
        ('p.T790M', 'THR-790MET'),
        ('p.T790M', 'THR-790 -> MET'),
        ('p.T790M', 'THR-790 --> MET'),
        ('p.T790M', 'THR(790)MET'),
        ('p.T790M', 'p.THR790MET'),

        # Protein variants - full amino acid names
        ('p.T790M', 'THR-to-MET substitution at position 790'),
        ('p.T790M', 'THR 790 is replaced by MET'),
        ('p.T790M', 'THR 790 mutated to MET'),
        ('p.T790M', 'THR 790 was mutated to MET'),
        ('p.T790M', 'THREONINE-to-METHIONINE mutation at residue 790'),
        ('p.T790M', 'THREONINE-to-METHIONINE mutation at amino acid 790'),
        ('p.T790M', 'THREONINE-to-METHIONINE mutation at amino acid position 790'),
        ('p.T790M', 'THREONINE-to-METHIONINE mutation at position 790'),
        ('p.T790M', 'THREONINE-to-METHIONINE mutation in position 790'),
        ('p.T790M', 'THREONINE-to-METHIONINE substitution at residue 790'),
        ('p.T790M', 'THREONINE-to-METHIONINE substitution at amino acid 790'),
        ('p.T790M', 'THREONINE-to-METHIONINE substitution at amino acid position 790'),
        ('p.T790M', 'THREONINE-to-METHIONINE substitution at position 790'),
        ('p.T790M', 'THREONINE-to-METHIONINE substitution in position 790'),
        ('p.T790M', 'THREONINE-to-METHIONINE alteration at residue 790'),
        ('p.T790M', 'THREONINE-to-METHIONINE alteration at amino acid 790'),
        ('p.T790M', 'THREONINE-to-METHIONINE alteration at amino acid position 790'),
        ('p.T790M', 'THREONINE-to-METHIONINE alteration at position 790'),
        ('p.T790M', 'THREONINE-to-METHIONINE alteration in position 790'),
        ('p.T790M', 'THREONINE-to-METHIONINE change at residue 790'),
        ('p.T790M', 'THREONINE-to-METHIONINE change at amino acid 790'),
        ('p.T790M', 'THREONINE-to-METHIONINE change at amino acid position 790'),
        ('p.T790M', 'THREONINE-to-METHIONINE change at position 790'),
        ('p.T790M', 'THREONINE-to-METHIONINE change in position 790'),
        ('p.T790M', 'THREONINE-to-METHIONINE at residue 790'),
        ('p.T790M', 'THREONINE-to-METHIONINE at amino acid 790'),
        ('p.T790M', 'THREONINE to METHIONINE mutation at residue 790'),
        ('p.T790M', 'THREONINE to METHIONINE mutation at amino acid 790'),
        ('p.T790M', 'THREONINE to METHIONINE mutation at amino acid position 790'),
        ('p.T790M', 'THREONINE to METHIONINE mutation at position 790'),
        ('p.T790M', 'THREONINE to METHIONINE mutation in position 790'),
        ('p.T790M', 'THREONINE to METHIONINE substitution at residue 790'),
        ('p.T790M', 'THREONINE to METHIONINE substitution at amino acid 790'),
        ('p.T790M', 'THREONINE to METHIONINE substitution at amino acid position 790'),
        ('p.T790M', 'THREONINE to METHIONINE substitution at position 790'),
        ('p.T790M', 'THREONINE to METHIONINE substitution in position 790'),
        ('p.T790M', 'THREONINE to METHIONINE alteration at residue 790'),
        ('p.T790M', 'THREONINE to METHIONINE alteration at amino acid 790'),
        ('p.T790M', 'THREONINE to METHIONINE alteration at amino acid position 790'),
        ('p.T790M', 'THREONINE to METHIONINE alteration at position 790'),
        ('p.T790M', 'THREONINE to METHIONINE alteration in position 790'),
        ('p.T790M', 'THREONINE to METHIONINE change at residue 790'),
        ('p.T790M', 'THREONINE to METHIONINE change at amino acid 790'),
        ('p.T790M', 'THREONINE to METHIONINE change at amino acid position 790'),
        ('p.T790M', 'THREONINE to METHIONINE change at position 790'),
        ('p.T790M', 'THREONINE to METHIONINE change in position 790'),
        ('p.T790M', 'THREONINE to METHIONINE at residue 790'),
        ('p.T790M', 'THREONINE to METHIONINE at amino acid 790'),
        ('p.T790M', 'THREONINE by METHIONINE at position 790'),
        ('p.T790M', 'THREONINE-790-METHIONINE'),
        ('p.T790M', 'THREONINE-790 -> METHIONINE'),
        ('p.T790M', 'THREONINE-790 --> METHIONINE'),
        ('p.T790M', 'THREONINE 790 METHIONINE'),
        ('p.T790M', 'THREONINE 790 changed to METHIONINE'),
        ('p.T790M', 'THREONINE-790 METHIONINE'),
        ('p.T790M', 'THREONINE 790-METHIONINE'),
        ('p.T790M', 'THREONINE 790 to METHIONINE'),
        ('p.T790M', 'THREONINE 790 by METHIONINE'),
        ('p.T790M', '790 THREONINE to METHIONINE'),
        ('p.T790M', 'METHIONINE for THREONINE at amino acid 790'),
        ('p.T790M', 'METHIONINE for THREONINE at position 790'),
        ('p.T790M', 'METHIONINE for THREONINE 790'),
        ('p.T790M', 'METHIONINE-for-THREONINE at position 790'),
        ('p.T790M', 'METHIONINE for THREONINE substitution at position 790'),
        ('p.T790M', 'METHIONINE-for-THREONINE substitution at position 790'),
        ('p.T790M', 'METHIONINE for a THREONINE at position 790'),
        ('p.T790M', 'METHIONINE for an THREONINE at position 790'),

        # Frameshift mutations
        ('p.T790fsX', 'T790fs'),
        ('p.T790fsX791', 'p.T790fsX791'),
        ('p.T790fsX791', 'p.THR790fsx791'),
        ('p.T790fsX791', 'THR790fsx791'),

        # Protein deletions
        ('p.790delT', 'THR790del'),
        ('p.790delT', 'p.T790del'),
        ('p.790delT', 'p.790delT'),
        ('p.790delT', 'T790del'),
        ('p.790delT', '790delT'),

        # DNA/cDNA variants - substitutions
        ('c.93G>A', 'c.93G>A'),
        ('c.93G>A', 'c.G93A'),
        ('c.93G>A', 'c.93G>A'),
        ('c.93G>A', 'c.93G/A'),
        ('c.93G>A', '93G>A'),
        ('c.93G>A', 'G/A-93'),
        ('c.93G>A', '93G->A'),
        ('c.93G>A', '93G-->A'),
        ('c.93G>A', 'G93->A'),
        ('c.93G>A', 'G93-->A'),
        ('c.93G>A', '93G-A'),
        ('c.93G>A', 'G modified A 93'),
        ('c.93G>A', '93G/A'),
        ('c.93G>A', '93,G/A'),
        ('c.93G>A', '(93) G/A'),
        ('c.93G>A', '93 (G/A)'),
        ('c.93G>A', 'G to A substitution at nucleotide 93'),
        ('c.93G>A', 'G to A substitution at position 93'),
        ('c.93G>A', 'G to A at nucleotide 93'),
        ('c.93G>A', 'G to A at position 93'),
        ('c.93G>A', 'g+93G>A'),

        # DNA/cDNA deletions
        ('c.93delG', 'c.93delG'),
        ('c.93delG', 'c.93Gdel'),
        ('c.93delG', '93delG'),
        ('c.93delG', '93Gdel'),

        # Multi-nucleotide substitutions
        ('c.93GGC>GAC', 'GGC93GAC'),

        # Range deletions
        ('c.93_94del', 'c.93-94del'),
        ('c.93_94del', 'c.93_94del'),
        ('c.93_94del', '93-94del'),
        ('c.93_94del', '93_94del'),

        # Duplications
        ('c.93dup', 'c.93dup'),
        ('c.93_94dup', 'c.93-94dup'),
        ('c.93_94dup', 'c.93_94dup'),
        ('c.93_94dup', '93-94dup'),
        ('c.93_94dup', '93_94dup'),

        # Genomic and mitochondrial variants
        ('g.93G>A', 'g.93G>A'),
        ('m.93G>A', 'm.93G>A'),
    ]

    # Remove all spaces from input
    mention = mention.replace(' ', '')

    # Try each pattern
    for pattern_out, pattern_in in examples:
        # Create regex from pattern by escaping then replacing placeholders
        regex = "^%s$" % re.escape(pattern_in.replace(' ', ''))

        # Define placeholder mappings for pattern variables
        mapping = [
            ('THREONINE', '(?P<from>Alanine|Cysteine|AsparticAcid|GlutamicAcid|Phenylalanine|Glycine|Histidine|Isoleucine|Lysine|Leucine|Methionine|Asparagine|Proline|Glutamine|Arginine|Serine|Threonine|Valine|Tryptophan|Tyrosine)'),
            ('METHIONINE', '(?P<to1>Alanine|Cysteine|AsparticAcid|GlutamicAcid|Phenylalanine|Glycine|Histidine|Isoleucine|Lysine|Leucine|Methionine|Asparagine|Proline|Glutamine|Arginine|Serine|Threonine|Valine|Tryptophan|Tyrosine)'),
            ('THR', '(?P<from>Ala|Arg|Asn|Asp|Cys|Glu|Gln|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val)'),
            ('MET', '(?P<to1>Ala|Arg|Asn|Asp|Cys|Glu|Gln|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val|X|\\*|Ter|Stop)'),
            ('790', '(?P<num>[1-9][0-9]*)'),
            ('791', '(?P<num2>[1-9][0-9]*)'),
            ('T', '(?P<from>[ABCDEFGHIKLMNPQRSTVWYZ])'),
            ('M', '(?P<to1>([ABCDEFGHIKLMNPQRSTVWYZX\\*]|stop))'),
            ('E', '(?P<to2>[ABCDEFGHIKLMNPQRSTVWYZX\\*])'),
            ('V', '(?P<to3>[ABCDEFGHIKLMNPQRSTVWYZX\\*])'),
            ('GGC', '(?P<from>[acgt]+)'),
            ('GAC', '(?P<to1>[acgt]+)'),
            ('G', '(?P<from>[acgt])'),
            ('A', '(?P<to1>[acgt])'),
            ('C', '(?P<to2>[acgt])'),
            ('93', '(?P<num>[\\+\\-]?[1-9][0-9\\-\\+]*)'),
            ('94', '(?P<num2>[\\+\\-]?[1-9][0-9\\-]*)')
        ]

        # Replace placeholders with unique temporary strings to avoid conflicts
        unique = {}
        for map_from, map_to in mapping:
            unique[map_from] = "!!!%04d" % len(unique)
            regex = regex.replace(map_from, unique[map_from])

        # Now replace temporary strings with actual regex patterns
        for map_from, map_to in mapping:
            regex = regex.replace(unique[map_from], map_to)

        # Try to match the pattern
        match = re.match(regex, mention, re.IGNORECASE)
        if match:
            # Extract matched groups and uppercase them
            d = {key: value.upper() for key, value in match.groupdict().items()}
            if 'num' in d:
                d['num'] = d['num'].rstrip('-+')

            # Format output based on pattern type
            if pattern_out == 'c.G>A':
                return "c.%s>%s" % (d['from'], d['to1'])
            elif pattern_out == 'c.93G>A':
                return "c.%s%s>%s" % (d['num'], d['from'], d['to1'])
            elif pattern_out == 'c.93delG':
                return "c.%sdel%s" % (d['num'], d['from'])
            elif pattern_out == 'c.GGC>GAC':
                return "c.%s>%s" % (d['from'], d['to1'])
            elif pattern_out == 'c.93GGC>GAC':
                return "c.%s%s>%s" % (d['num'], d['from'], d['to1'])
            elif pattern_out == 'c.93G>A,C':
                return "c.%s%s>%s,%s" % (d['num'], d['from'], d['to1'], d['to2'])
            elif pattern_out == 'c.93_94del':
                return "c.%s_%sdel" % (d['num'], d['num2'])
            elif pattern_out == 'c.93_94dup':
                return "c.%s_%sdup" % (d['num'], d['num2'])
            elif pattern_out == 'c.93dup':
                return "c.%sdup" % d['num']
            elif pattern_out == 'g.93G>A':
                return "g.%s%s>%s" % (d['num'], d['from'], d['to1'])
            elif pattern_out == 'm.93G>A':
                return "m.%s%s>%s" % (d['num'], d['from'], d['to1'])
            elif pattern_out == 'p.TM':
                return "p.%s%s" % (AMINO_ACID_MAP[d['from']], AMINO_ACID_MAP[d['to1']])
            elif pattern_out == 'p.T790M':
                return "p.%s%s%s" % (AMINO_ACID_MAP[d['from']], d['num'], AMINO_ACID_MAP[d['to1']])
            elif pattern_out == 'p.T790M/E':
                return "p.%s%s%s,%s" % (AMINO_ACID_MAP[d['from']], d['num'], AMINO_ACID_MAP[d['to1']], AMINO_ACID_MAP[d['to2']])
            elif pattern_out == 'p.T790M/E/V':
                return "p.%s%s%s,%s,%s" % (AMINO_ACID_MAP[d['from']], d['num'], AMINO_ACID_MAP[d['to1']], AMINO_ACID_MAP[d['to2']], AMINO_ACID_MAP[d['to3']])
            elif pattern_out == 'p.T790fsX':
                return "p.%s%sfsX" % (AMINO_ACID_MAP[d['from']], d['num'])
            elif pattern_out == 'p.T790fsX791':
                return "p.%s%sfsX%s" % (AMINO_ACID_MAP[d['from']], d['num'], d['num2'])
            elif pattern_out == 'p.790delT':
                return "p.%sdel%s" % (d['num'], AMINO_ACID_MAP[d['from']])

    return None
