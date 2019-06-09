rm -f TIC3.list
#ls -1 /pike/pipeline/TIC3/tess_*_1_1_*.h5 > TIC3.list
#ls -1 /pike/pipeline/TIC3/tess_*_1_2_*.h5 > TIC3.list
echo "FILE" > TIC3.list
for i in `seq 26`
do
    for j in `seq 4`
    do
	for k in `seq 4`
	do
	    echo "$i"_"$j"_"$k"
	    ls -1 /pike/pipeline/TIC3/tess_*_"$i"_"$j"_"$k".h5 >> TIC3.list
	done
    done
done
