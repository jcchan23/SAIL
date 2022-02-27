#!/usr/bin/perl -w

open(FILE,"$ARGV[0].sc")||die;
@fn=<FILE>;
$f=@fn;
$id=$ARGV[0];
for($i=0;$i<$f;$i++)
{
    @w=split /\s+/,$fn[$i];
    #$w=@w;
    for($j=1;$j<@w;$j++)
    {
        if($w[$j]>8){$w[$j]=8;}if($w[$j]<-8){$w[$j]=-8;}
        push @a,$w[$j];
    }
}
close(FILE);
@max=sort{$b<=>$a}@a;$max=$max[0];$min=$max[@max-1];
$dis=$max-$min;
#print "$max $min $dis\n";exit;
if($min eq "" or $max eq ""){print "$ARGV[0] $min $max $dis\n";exit;}

open NEW,">$id.sc.nml";
open FN,"$id.sc";#$suffix=$ARGV[0];
@fn=<FN>;
$f=@fn;
for($i=0;$i<$f;$i++)
{
    @w=split /\s+/,$fn[$i];
    $w=@w;
    for($j=1;$j<$w;$j++)
    {$new[$i][$j]=((($w[$j]-$min)/$dis)*2)-1;#print "$new[$i][$j] $w[$j] $min\n";exit;#
        printf NEW"%4.3f ",$new[$i][$j];
    }print NEW "\n";
}
