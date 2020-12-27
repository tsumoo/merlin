## DNN統計的音声合成ツールキット Merlin の中身を理解をする
https://r9y9.github.io/blog/2017/08/16/trying-to-understand-merlin/
を読む。

* デモスクリプトではrun_merlin.pyに用途に応じた設定ファイルを与えることで、継続長モデルの学習/音響モデルの学習/パラメータ生成など、音声合成に必要なステップを実現しています。

* HTSと同様に、frontend, backendといった部分は提供していません。Merlinの論文にもあるように、HTSの影響を受けているようです。
* Text-processing (Frontend) 英語ならFestival
デモスクリプトでは、Frontendによって生成されたフルコンテキストラベル（HTS書式）が事前に同梱されているので、Festivalをインストールする必要はありません。 
 misc以下に、Festivalを使ってフルコンテキストラベルを作るスクリプト (make_labels) がある
* Speech analysis/synthesis (Backend) WORLD(推奨)やSTRAIGHT

Merlinのスクリプトによってはかれるデータは、基本的に

x.astype(np.float32).tofile("foobar.bin")
といった感じで、32bit浮動小数点のnumpyの配列がヘッダなしのバイナリフォーマットで保存されています。デバッグ時には、

np.fromfile("foobar.bin", dtype=np.float32)
として、ファイルを読み込んでインスペクトするのが便利

## MerlinのTensorFlowDNN構造まとめ
http://k17trpsynth.hatenablog.com/entry/2018/03/06/212332



## Windowsのvscodeでテスト
GPUが使えない。

### 環境構築
docker内で実行。reopen し、 C++を選択。.devcontainer以下が自動生成される

#python2.7 -> 3.5
#https://linuxconfig.org/how-to-change-default-python-version-on-debian-9-stretch-linux
#apt update
#apt install csh realpath autotools-dev automake python-pip3
#pip3 install -r requirements.txt
#ただしnumpyは1.13.1 
#https://github.com/CSTR-Edinburgh/merlin/issues/466
#scipyは1.2.3
#https://github.com/CSTR-Edinburgh/merlin/issues/470

### build tools

http://jrmeyer.github.io/tts/2017/02/14/Installing-Merlin.html
の手順で実行

cd tools
./compile_tools.sh

すると途中でエラー。-lSPTKに失敗。以下で対処。
cp lib/libSPTK.a ../postfilter/src/SPTK/lib/


apt install python-pip
pip install numpy  ### 先にnumpyを入れないとbandmatが入らない
pip install scipy matplotlib lxml theano bandmat

merlin/egs/slt_arctic/s1$ ./run_demo.sh


synthesized audio files are in: experiments/slt_arctic_demo/test_synthesis/wav
All successfull!! Your demo voice is ready :)

run_full_voice.shを実行してみたが、学習に時間がかかりすぎる。


### on gitpod

.gitpod.Dockerfileなどをpushしないといけない。そのため自分のところにforkしてpush
いろいろやってみたが、GPUが使えなさそう。
python 3.7だとインストールできないものもある
https://stackoverflow.com/questions/61382768/python3-7-mlpy-installation-error-pythreadstate-aka-struct-ts-has-no-memb

### on colab


