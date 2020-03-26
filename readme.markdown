This is the implementation of CVPR2020 paper "Stereoscopic Flash and No-Flash Photography for Shape and Albedo Recovery".
# Introduction
We adopt a stereo camera and a flashlight, take three images, and recover high-quality shape and albedo. Three images construct two image pairs: a no-flash stereo image pair and a flash/no-flash pair.
From the stereo pair, we obtain coarse shape and normal map. From the flash/no-flash image pair, we recover the high-frequency details missing in the coarse shape. Further, an albedo map is computed from the flash/no-flash pair up to flashlight intensity scale.
![pipeline](/img/pipeline.png) 
# Usage
The code was tested on Ubuntu 16.04 and MacOS 10.15 with Python 3.6.

1. Download data from [this link](https://drive.google.com/open?id=1kERor3ToBMs1LbGFv4X_IDbGvPc-XVDe), and extract it in the root directory.

2. Ensure all required packages are installed:
`pip3 install -r requirements.txt`

3. Run `main.py`, and check results in `results/[execution time]`.

4. You might want to tune parameters. You can modify `options`, a python dictionary comprising all tunable parameters, defined in `main.py`.

# Citation
If you used this code in your publication, please consider citing the following paper:
```
@inproceedings{cao2020stereoscopic,
title = {Stereoscopic Flash and No-Flash Photography for Shape and Albedo Recovery},
author = {Xu Cao and Michael Waechter and Boxin Shi and Ye Gao and Bo Zheng and Yasuyuki Matsushita},
year = {2020},
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```
# Contact
For any questions/comments/bug reports, please feel free to contact cao.xu@ist.osaka-u.ac.jp