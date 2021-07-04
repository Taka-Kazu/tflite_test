# Requirement
- Docker (>=19.03)
- docker-compose

# Setup
- Download [pretrained model](https://github.com/google-coral/test_data/raw/master/ssdlite_mobiledet_coco_qat_postprocess.tflite) and [label file](https://github.com/google-coral/test_data/raw/master/coco_labels.txt) to the root of this project

- Prepare video file or device (`python main.py -h` for detail)

- Build and run docker container
```
$ make build
$ make run
# in docker container
$ python main.py
```
