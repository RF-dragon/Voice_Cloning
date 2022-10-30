# Voice Cloning

([简体中文](README.md) | English)

### Description

This program is a voice cloning program that is improved based on the PaddleSpeech voice cloning demo. This program uses Spleeter for human voice extraction, which allows a better feature extraction.

### Installation
1. Clone or download this repository.
2. Install Python. Please make sure that you have added Python to PATH.
3. Install PaddlePaddle and PaddleSpeech on https://www.paddlepaddle.org.cn/en according to your OS and CUDA version. If you are unfamiliar with CUDA, please select "CPU" on the "Compute Platform" column.
4. Install NumPy, Soundfile, Spleeter and Yacs with the following command:
```shell
python -m pip install numpy soundfile spleeter yacs -i https://pypi.tuna.tsinghua.edu.cn/simple -U
```

### Run
1. Put audio files to be cloned into the [input](input) folder.
2. Run [Voice Cloning.py](Voice%20Cloning.py). The program will split human voices and background noises with Spleeter. A feature file that ends with ".voice" will be generated for each input files under the [voice library](voice%20library) folder.
3. Enter the text to be synthesized in [text.txt](text.txt).
4. Run [TTS.py](TTS.py). The program will generate a synthesized audio file for each feature file inside the [output](output) folder.

The feature extraction and the Text to Speech will take about half a minute respectively. Please wait patiently.

### References
Hennequin, R., Khlif, A., Voituret, F., & Mousallam, M. (2020). Spleeter: a fast and efficient music source separation tool with pre-trained models. _Journal of Open Source Software_, _5_(50) 2154. The Open Journal. https://doi.org/10.21105/joss.02154

Zhang, H., Yuan, T., Chen, J., Li, X., Zheng, R., Huang, Y., Chen, X., Gong, E., Chen, Z., Hu, X., Yu, D., Ma, Y., & Huang, L. (2022). PaddleSpeech: An Easy-to-Use All-in-One Speech Toolkit. _Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Demonstrations_.

Zheng, R., Chen, J., Ma, M., & Huang, L. (2021). _International Conference on Machine Learning_, 12736-12746. PMLR.