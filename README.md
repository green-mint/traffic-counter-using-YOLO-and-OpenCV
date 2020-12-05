# Traffic Detection and Counting Using YOLOv3

## Video Visualization Demo

<img src="demo/traffic_counter.gif" alt="demo video">

## Data:
The data has been obtained from a [youtube](https://www.youtube.com/watch?v=wqctLW0Hb_0&ab_channel=AndreyNikishaev) video. Click [here](https://redirector.googlevideo.com/videoplayback?expire=1607124198&ei=hnDKX5WxMqHIkwa5wYHQBA&ip=168.235.107.39&id=o-ANc_KJWH0osf1q4RGFGy2x-o3onVGBKhVvC0JcnuUzlz&itag=18&source=youtube&requiressl=yes&mh=qV&mm=31%2C26&mn=sn-a5mlrnes%2Csn-nx57ynlz&ms=au%2Conr&mv=m&mvi=2&pl=23&initcwndbps=642500&vprv=1&mime=video%2Fmp4&ns=U7RoUrI5LAco3r3Sq8mMXZIF&gir=yes&clen=107741911&ratebypass=yes&dur=2048.092&lmt=1554143545151518&mt=1607102201&fvip=2&c=WEB&txp=5431432&n=UgKjijX5a5mBq3PDD&sparams=expire%2Cei%2Cip%2Cid%2Citag%2Csource%2Crequiressl%2Cvprv%2Cmime%2Cns%2Cgir%2Cclen%2Cratebypass%2Cdur%2Clmt&sig=AOq0QJ8wRAIgHCG8oY94d2YJ65gb1J_TJDvZVpdalTevgMB76Rk4xmsCIEpOYiXN423cISsX3ETanulzSS1lBnqbdPxJbakoA6NA&lsparams=mh%2Cmm%2Cmn%2Cms%2Cmv%2Cmvi%2Cpl%2Cinitcwndbps&lsig=AG3C_xAwRgIhAM-8C_8E_WauuIIHM-AWIpiKkC5_SpNSNxZPWfP50-6VAiEAqXOnIF2f6TA4D8W0u2XvF0BtpgbhIfLTHjTZV54VeSs%3D&title=Road+traffic+video+for+object+recognition) to download.

Due to the sheer size of the file I had to trim it down to approximately 30 sec to obtain results faster.

## Implementation Guide:

1. Clone the repo and cd into it
  ```
  $ git clone https://github.com/green-mint/traffic-counter-using-YOLO.git
  $ cd traffic-counter-using-YOLO
  ```
2. Download [yolov3.weights](https://www.dropbox.com/s/99mm7olr1ohtjbq/yolov3.weights?dl=0) and move them into the `yolov3` directory. Your directory structure should now be similar to
```
├── demo
│   └── traffic_counter.gif
├── input.mp4
├── README.md
├── requirements.txt
├── tracker.py
├── traffic_main.py
├── utils.py
└── yolov3
    ├── coco.names
    ├── yolov3.cfg
    └── yolov3.weights
```

3. Create a new python virtual environment and install the required libraries by running the following commands
  ```
  $ python3 -m venv path/to/virtual/environment
  $ source path/to/virtual/environment/bin/activate
  $ pip install -r requirements.txt
  ```

4. Run the `traffic-main.py` script to test your input video
  ```
  $ python traffic-main.py --input/-i path/to/input/clip.mp4 --output/-o path/to/output/clip.avi
  ```
  Additional optional arguments are `--confidence/-c` and `--threshold/-t` with `default=0.5` and `default=0.3` respectively

## Citations: 
### Yolo:
```
@article{redmon2016yolo9000,
  title={YOLO9000: Better, Faster, Stronger},
  author={Redmon, Joseph and Farhadi, Ali},
  journal={arXiv preprint arXiv:1612.08242},
  year={2016}
}
```
### Sort:
```
@inproceedings{Bewley2016_sort,
  author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
  booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
  title={Simple online and realtime tracking},
  year={2016},
  pages={3464-3468},
  keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
  doi={10.1109/ICIP.2016.7533003}
}
```
