#!/usr/bin/perl -w

unless (@ARGV){print "perl $0 pdbID/strucID\n";exit;}
$id  =  $ARGV[0];
open(PDB,"$id.pdb")||die"$id.pdb is not exist\n";
@pdb = <PDB>;
&check_bb(@pdb);
open NEW,">$id.SALA_bb";

if($ARGV[1]){
	$chain_select = $ARGV[1];
}
else{$chain_select = "NULL";}

for(@pdb)
{
	if($_ =~m/^ATOM/)
	{
		$atomtype = substr($_,13,3);
		$atomtype =~s/ //g;chomp$_;
		$chain = substr($_,21,1);
		if($chain ne  $chain_select and  $chain_select eq "NULL"){ $chain_select = $chain;}
		if($chain eq $chain_select){
			if($atomtype eq "C"){$seq = &change_restype($_);print NEW"$seq\n";}
			if($atomtype eq "CA"){$seq = &change_restype($_);print NEW"$seq\n";}
			if($atomtype eq "N"){$seq = &change_restype($_);print NEW"$seq\n";}
			if($atomtype eq "O"){$seq = &change_restype($_);print NEW"$seq\n";}

		}
	}
}

system "./bin/addCB $id.SALA_bb >$id.ALA.tt";
open(TT,"$id.ALA.tt");
open NEW,">$id.ALA.pdb";
for(<TT>)
{
	if($_ =~m/TER/){last;}
	$a = substr($_,0,54);
	$more = "  1.00100.00           ";
	$at = substr($_,13,1);
	print NEW $a."$more"."$at"."  \n";
}
system "./bin/check_rewritepdb.pl $id.ALA.pdb >$id.tt";
system "mv $id.tt $id.ALA.pdb";
sub change_restype
{
	my $data = $_[0];
	my $part1 = substr($data,0,17);
	my $res = "ALA";
	my $part2 = substr($data,20,);
	my $newseq = $part1.$res.$part2;
	return $newseq;
}

unlink "$id.ALA.tt";
unlink "$id.SALA_bb";
sub check_bb
{
	my @pdb = @_;
	my %bbatom;
	my $index;
	for(@pdb)
	{
		if($_=~m/^ATOM/){
            $chain = substr($_,21,1);
            $count_chain{$chain} = $chain;
			$index = substr($_,22,4);
			$index =~s/\s+//g;
			my $AA_type = substr($_,17,3);
			my $atomtype = substr($_,13,3);
			$atomtype=~s/\s+//g;
			#$index = $index.$atomtype;
			if($AA_type ne "GLY"){$bbatom{$index}[1]="M";}
			else{$bbatom{$index}[1]="X";}
			if($atomtype eq "CA" or $atomtype eq "C" or $atomtype eq "N" or $atomtype eq "O"){
				$bbatom{$index}[0]++;
                #print "$index $bbatom{$index}[0]\n";
			}	
			if($AA_type ne "GLY" and $atomtype eq "CB"){
				$bbatom{$index}[0]++;	
			}
		}
	}
    $chain_no = (keys %count_chain);
    if( $chain_no  >1){ print "<p>There are $chain_no chains in the given structure</p><p>Please input monomer structure</p>";exit 1;}
	my $resindex;
	for(sort{$a<=>$b}(keys%bbatom))
	{
		$resindex=$_;
		#if(!defined($bbatom[$_][1])){print "$resindex not exit\n";exit;}
		if($bbatom{$resindex}[1] eq "M"){
			if($bbatom{$resindex}[0] <5){
				print "<p>Residue $resindex only has $bbatom{$resindex}[0] backbone atoms and lack backbone atom</p><p>The program will stop. Please input a structure file with complete backbone atoms</p>";
                exit 1;# set it as error so that the shell can takes the info, in this case not to run other program
			}
		}
		else{
			if($bbatom{$resindex}[0] <4){
				print "<p>Residue $resindex only has $bbatom{$resindex}[0] backbone atoms and lack backbone atom</p><p>The program will stop. Please input a structure file with complete backbone atoms</p>";
                exit 1;# same
			}
		}
	}

}

