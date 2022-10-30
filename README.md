# Voice Cloning

（简体中文 | [English](README.en.md)）

### 介绍
本程序为基于PaddleSpeech的语音克隆demo改进的语音克隆程序。本程序使用了Spleeter进行人声分离，并根据分离出的人声进行特征提取，因此能得到性能更好的语音特征。

### 安装
1. 将仓库克隆或下载到本地。
2. 安装Python。请确保在安装过程中将Python添加到PATH。
3. 访问[PaddlePaddle](https://www.paddlepaddle.org.cn/)，根据自己的操作系统和CUDA版本安装PaddlePaddle和PaddleSpeech。如果你对CUDA不熟悉，请在“计算平台”一栏中选择“CPU”。
4. 使用以下命令安装NumPy、Soundfile、Spleeter和Yacs：
```shell
python -m pip install numpy soundfile spleeter yacs -i https://pypi.tuna.tsinghua.edu.cn/simple -U
```

### 运行
1. 将待克隆的音频文件放入[input](input)文件夹下。
2. 运行[Voice Cloning.py](Voice%20Cloning.py)。程序会在[input](input)文件夹下使用Spleeter将人声和环境噪声分离，随后每一个待克隆的音频会在[voice library](voice%20library)文件夹中生成一个以“.voice”结尾的音频特征文件。
3. 在[text.txt](text.txt)中输入待合成的文本。
4. 运行[TTS.py](TTS.py)。每一个[voice library](voice%20library)文件夹中的音频特征文件将在[output](output)文件夹中合成一段音频。

提取和语音合成大约会分别花费半分钟，请耐心等待。

### 参考文献
Hennequin, R., Khlif, A., Voituret, F., & Mousallam, M. (2020). Spleeter: a fast and efficient music source separation tool with pre-trained models. _Journal of Open Source Software_, _5_(50) 2154. The Open Journal. https://doi.org/10.21105/joss.02154

Zhang, H., Yuan, T., Chen, J., Li, X., Zheng, R., Huang, Y., Chen, X., Gong, E., Chen, Z., Hu, X., Yu, D., Ma, Y., & Huang, L. (2022). PaddleSpeech: An Easy-to-Use All-in-One Speech Toolkit. _Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Demonstrations_.

Zheng, R., Chen, J., Ma, M., & Huang, L. (2021). _International Conference on Machine Learning_, 12736-12746. PMLR.