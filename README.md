# Training neural network for object detection

clone URL
https://github.com/felicepantaleo/ML-experiments
```
pip3 install pyqt5  # pyqt5 can be installed via pip on python3 
pip3 install labelme
```
if installation is succesful run:
```
labelme
```
click on open dir and then for each image cret a new segmantation around the object to be detected. Then click on save. This will generate a `json` file for each image. 
In ML-experiments run:
```
python3 convert_labelme_to_coco.py <imagefolder>
```
this will create a `trainval.json` file in the coco dataformat for the entire dataset.


