# FasterRCNN_Document_Segmentation
Deep learning networks for object detection to the task of optical character recognition in order to build image features taylored for documents. In contrast to scene text reading in natural images using networks pretrained on ImageNet, our document reading is performed with small networks inspired by MNIST digit recognition challenge, at a small computational budget and a small stride. The object detection modern frameworks allow a direct end-to-end training, with no other algorithm than the deep learning and the non-max-suppression algorithm to filter the duplicate predictions. The trained weights can be used for higher level models, such as, for example, document classification, or document segmentation.



What we need to create is the following. Start by creating all of the empty folders.
~~~
+VOCdevkit
    +VOC2012
        +Annotations
                -A bunch of .xml labels
        +JPEGImages
                -A bunch of .jpg images
        +ImageSets
                +Main
                        -aeroplane_trainval.txt (This is just a list of the jpeg files without file extensions, the train.py script reads this file for all the images it is supposed to include.
                        -trainval.txt (An exact copy of the aeroplane_trainval.txt)

        +trainingConfig.config (training config file similar to https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)
        +Originals
                      -all your original image files (just for easy access)

~~~
steps:-

step-1: genius@Genius:/var/www/html/Reinforcement/CNN/FasterRCNNTutorial-master$ cd models/research

step-2: genius@Genius:/var/www/html/Reinforcement/CNN/FasterRCNNTutorial-master/models/research$ /home/data/Desktop/final_project/project_ocr/protoc_3.3/bin/protoc object_detection/protos/*.proto --python_out=.

step-3: genius@Genius:/var/www/html/Reinforcement/CNN/FasterRCNNTutorial-master/models/research$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

step-4: python object_detection/dataset_tools/create_pascal_tf_record.py -h

step-5: python object_detection/dataset_tools/create_pascal_tf_record.py --data_dir=/home/data/Desktop/final_project/project_ocr/project_segmentation/VOCtemplate --year=VOC2012 --output_path=/home/data/Desktop/final_project/project_ocr/project_segmentation/pascal.record --label_map_path=/home/data/Desktop/final_project/project_ocr/project_segmentation/label.pbtxt --set=trainval


step-6: python object_detection/legacy/train.py --train_dir=/home/data/Desktop/final_project/project_ocr/project_segmentation/train --pipeline_config_path=/home/data/Desktop/final_project/project_ocr/project_segmentation/faster_rcnn_resnet101_coco.config
