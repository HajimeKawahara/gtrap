9/3 改善ISSUES(atode)
散乱光除去アルゴリズムの再考。たとえば線形和で係数をフィットするとか、、、

f_target(lam) - sum_i a_i f_i(lam)

---------------------------------------------
9/3 Double pulse探査 (D?)

おそらく、single pulse(SP)はたとえ真であったとしても、
追観測に自信がもてるほどのライトカーブは得られないのではないか？
（SPはFPもかなり多い）
で、あればdouble pulseサーチalgorithmをつくる

- algorithm
TLSのpeak pair: (peak[0] and [1])
もしnこ許すなら、nC2通りのpeak pair(PP)をつくる
これを学習(peak pairはchannelでよい？）

- pickdplc
- mockdplc

- dnn/dpnet

+ DPのFPの多くはBIPなのでCSベース＋astのwindow sizeをmiddleにしてデータ構成を作りなおす

2019
---------------------------------------------



---------------------------------------------
dnn/pulsenetから引き継ぎ
Single Pulse

9/2
checking by 1

CAは除去率がさがる。アステロイドシグナルのwindowが適切でないかもしれない。

CA: LC(wide), LC(local), BIP, AST 
CS: LC(wide), LC(local), BIP 


9/1
checking by picked data... to 32
あとたぶんデータ拡張

8/31
TESTはBIP, asteroidに関しては個別にテストすべき。ータがもっと必要。

train
A: LC(wide,local)+asteroid indicator

B: LC(wide,local)+asteroid indicator+inverse cross background
Bs: B but separableConv1D


2019
-----------------------------------------------------------------