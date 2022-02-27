#!/usr/bin/perl -w

#check ASER BSER and rewrite the pdb aa and atom index from 1
open(PDB,"$ARGV[0]")||die"not $ARGV[0]\n";
@pdb=<PDB>;
@newpdb=&checkDoubleAtom(@pdb);
@newpdb1=&rewritePDB(@newpdb);
for(@newpdb1){print "$_\n";}exit;
sub checkDoubleAtom
{
	my @line=@_;my @new;
	for(@line)
	{
		if($_=~m/^ATOM/)
		{#print "$_\n";exit;
			my $line=$_;chomp $line;#$s=length $line;
			my $AA=substr($line,16,4);
			$AA=~s/\s+//g;
			#my $insertion=substr($_,26,1);
			#if($insertion ne " "){print "insert $line $ARGV[0]\n";exit;}
			my $l=length $AA;
			if($l==4)
			{#print "$AA\n";
				my $pos=substr($_,22,4);$pos=~s/\s+//g;$count{$pos}++;
				if($count{$pos}==1){ $tag{$pos}=$AA;}
				if($AA eq $tag{$pos}){$before=substr($line,0,16);$after=substr($line,20);
					$newline="$before"."$AA"."$after";push @new,$newline;}
			}
			else{push @new,$line;}#print "$line\n";}
	}
}
return (@new);
}

sub rewritePDB
{
	my @line=@_;my $l=@line;
	my $before=substr($line[0],11,11);
	my $after=substr($line[0],26,);
	my @pdb1;
	my $atomindex=1;
	my $info="ATOM  "."    1"."$before"."   1"."$after";
	push @pdb1,$info;$k=1;#print "$l\n";exit;
	for($i=1;$i<$l;$i++)
	{	
		chomp $line[$i];
		$before=substr($line[$i],11,11);
		my $after=substr($line[$i],26,);
		$nowchain=substr($line[$i],21,1);
		$previouschain=substr($line[$i-1],21,1);
		$aaindex=substr($line[$i],22,5);
		$aaindex=~s/\s+//g;$aaprevious=substr($line[$i-1],22,5);$aaprevious=~s/ //g;#consider insert

			if($nowchain eq $previouschain)
			{if ($aaindex=~m/\w/){
						     if($aaindex eq $aaprevious)
						     {$atomindex++;$atomindex=&format($atomindex,5);
							     $no=&format($k,4);
							     $info="ATOM  "."$atomindex"."$before"."$no"."$after";push @pdb1,$info;
						     }
						     else{$atomindex++;$atomindex=&format($atomindex,5);
							     $k++;$no=&format($k,4);$info="ATOM  "."$atomindex"."$before"."$no"."$after";
							     push @pdb1,$info;
						     }

					     }
			else{
				if($aaindex==$aaprevious)
				{
					$no=&format($k,4);
					$info="$before"."$no"."$after";push @pdb1,$info;#print"$after\n";#exit;
				}
				else{
					$k++;	$no=&format($k,4);$info="$before"."$no"."$after";push @pdb1,$info;
				}
			}
			}
			else{
				$k=1;
				$no=&format($k,4);
			}
	}#print "@pdb1\n";
	return (@pdb1);
}


sub format
{
	my $input=$_[0];my $length=$_[1];
	my $l=length $input;#print "$l $input\n";exit;
	my $s=$length-$l;
	my $output="";my $i;
	for($i=0;$i<$s;$i++)
	{
		$output=$output." ";
	}
	$output=$output.$input;
	return($output);
}
