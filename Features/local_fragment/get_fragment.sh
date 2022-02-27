#!/bin/bash

inputdir=$(dirname $1)
outputdir=$2

filename=$(basename $1 .pdb)

# Make copy of input file
cp $1 ./${filename}_t.pdb || exit 1

# Extract backbone, get_ala_backbone.pl for next script, get_seq_backbone.pl for the last
perl ./bin/get_ala_backbone.pl ${filename}_t || exit 2
perl ./bin/get_seq_backbone.pl ${filename}_t || exit 3

# Generate fragments
echo -e "\"${filename}_t.ALA.pdb\"\\t`grep CA ${filename}_t.ALA.pdb | wc -l`" > ${filename}_t.list
#  Usage: main libdir/ liblist querydir list
./bin/featuregen_fragments ./data/db/ ./data/db_list ./ ./${filename}_t.list || exit 4
perl ./bin/get_SP.pl ${filename}_t || exit 5

# Generate rotomers
# Usage: RUN dfirelib1 sdir pdb
./bin/featuregen_rotomers ./dfirelib1 ./ ${filename}_t.ALA.pdb > ${filename}_t.sc || exit 6
perl ./bin/get_SC.pl ${filename}_t || exit 7
cat ${filename}_t.sc.nml | tr -s " " | cut -d " " -f 2-27,29-114 > ${filename}_t.features.rotomers || exit 8

# Extract atom positions
# Print file     extract columns            separate squashed fields                                                         del spaces  remove CB    del start spaces  del trailing sp   replace N      replace CA      replace C      replace O      write to file
# cat ${filename}_t.ALA.pdb | cut -c 13-16,23-26,31-54 | sed 's:^\(.\{4\}\)\(.\{4\}\)\(.\{8\}\)\(.\{8\}\)\(.\{8\}\).*$:\1 \2 \3 \4 \5:' | tr -s " " | grep -v CB | sed 's:^[ ]*::' | sed 's:[ ]*$::' | sed 's:N:0:' | sed 's:CA:1:' | sed 's:C:2:' | sed 's:O:3:' > ${filename}_t.positions || exit 9

mv ${filename}_t.features.fragments ${outputdir}/${filename}.fragments
mv ${filename}_t.features.rotomers ${outputdir}/${filename}.rotomers
mv ${filename}_t.SEQ.pdb ${outputdir}/${filename}_fragroto.pdb

# Clean things up
rm -f ${filename}_t* || exit 10


