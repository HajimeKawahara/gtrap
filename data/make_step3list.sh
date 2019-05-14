rm -f step3.list
#ls -1 /pike/pipeline/step3/tess_*_1_1_*.h5 > step3.list
#ls -1 /pike/pipeline/step3/tess_*_1_2_*.h5 > step3.list
for i in `seq 24`
do
    for j in `seq 4`
    do
	for k in `seq 4`
	do
	    echo "$i"_"$j"_"$k"
	    ls -1 /pike/pipeline/step3/tess_*_"$i"_"$j"_"$k".h5 >> step3.list
	done
    done
done
