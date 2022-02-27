#!/usr/bin/perl -w

$id=$ARGV[0];
open(SP,"$id.sp")||die;
@sp=<SP>;
open NEW,">$id.features.fragments";
#shift @sp;
$max=99;
$min=0;
$dis=$max-$min;
for(@sp)
{
    if($_=~m/\#/){next;}
    $_=~s/^\s+//g;
    @w=split /\s+/,$_;
    for($i=1;$i<21;$i++)
    {
        $new=2*(($w[$i]-$min)/$dis)-1;
        printf NEW"%4.3f ",$new;
    }
    print NEW"\n";
}
