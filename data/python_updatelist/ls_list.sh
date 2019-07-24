listname=ctl.list
directory1=/manta/pipeline/CTL1
directory2=/stingray/pipeline/CTL1Z


rm -f "$listname"
echo "FILE" > "$listname"

#sector
for i in `seq 10`
do
#camera    
    for j in `seq 4`
    do
#ccd
	for k in `seq 4`
	do
	    echo "$i"_"$j"_"$k"
	    ls -1 "$directory1"/tess_*_"$i"_"$j"_"$k".h5 >> "$listname"
	done
    done
done
for i in `seq 11 26`
do
    for j in `seq 4`
    do
	for k in `seq 4`
	do
	    echo "$i"_"$j"_"$k"
	    ls -1 "$directory2"/tess_*_"$i"_"$j"_"$k".h5 >> "$listname"
	done
    done
done
