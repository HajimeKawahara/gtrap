import numpy as np
import os
Nbatch=32 # bacth num for an execute.
Neach = 1000 # num for each
Nsh = 207 # num of eachshells
# total light curves = Nbatch*Nsh

python_execute="../gtls_slctess.py"
dirname="/home/kawahara/gtrap/examples/sh"
allname="mockall_slc.sh"
eachname="mockeach_slc"


def make_mockall_sh():
    #execute this sh for all shells
    
    f=open(os.path.join(dirname,allname),"w")
    f.write("for i in `seq 1 "+str(Nsh)+"`"+"\n")
    f.write("do"+"\n")
    f.write('echo "$i";'+"\n")
    f.write(os.path.join(dirname,eachname)+'"$i".sh &> log"$i" &'+"\n")
    f.write("done"+"\n")
    f.close()      

def make_mock_each():
    
    for i in range(1,Nsh+1):
        filename=eachname+str(i)+".sh"
        f=open(os.path.join(dirname,filename),"w")        
        #injected samples
        ex = Neach*(i)+int(np.random.rand()*Neach)+1
        f.write('i='+str(ex)+';'+"\n")
        f.write('a='+str(Nbatch)+';'+"\n")
        f.write('s=$(($a * $i));'+"\n")
        f.write('e=$(($s + $a - 1));'+"\n")
        f.write('echo "$i $s $e";'+"\n")
        f.write('python '+python_execute+' -i $s -j $e -fig -n 2;'+"\n")

        #no injected samples
        exn = Neach*(i)+int(np.random.rand()*Neach)+1        
        f.write('j='+str(exn)+';'+"\n")
        f.write('l=$(($a * $j));'+"\n")
        f.write('k=$(($l + $a - 1));'+"\n")
        f.write('echo "$j $l $k";'+"\n")
        f.write('python '+python_execute+' -i $l -j $k -fig -n 2 -q;'+"\n")

        f.close()
    
if __name__ == "__main__":
    make_mock_each()
    make_mockall_sh()
