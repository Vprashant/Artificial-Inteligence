# FasterRCNN_Document_Segmentation
Deep learning networks for object detection to the task of optical character recognition in order to build image features taylored for documents. In contrast to scene text reading in natural images using networks pretrained on ImageNet, our document reading is performed with small networks inspired by MNIST digit recognition challenge, at a small computational budget and a small stride. The object detection modern frameworks allow a direct end-to-end training, with no other algorithm than the deep learning and the non-max-suppression algorithm to filter the duplicate predictions. The trained weights can be used for higher level models, such as, for example, document classification, or document segmentation.

## pre-requsites:
* cloned and built the tensorflow/models/research folder into the tensorflow directory, you may not need to run the build files which are included with this. If you get script not found errors from the python commands then try running the various build scripts. (https://github.com/tensorflow/models/tree/master/research) 
* Jupyter notebook (pip install --user jupyter) 
* labelimg https://github.com/tzutalin/labelImg or pip install labelImg

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

Copy all your training images into the 'Originals' folder
Copy the set of images that you want to train on into the JPEGImages folder
Create image tiles if needed (See 'How big should my images be?')
Resize them to X*600 (See 'How big should my images be?')

~~~
cd .../JPEGImages
for file in $PWD/*.jpg
do
convert $file -resize 717x600 $file
done
~~~

Optionally, rename them to consecutive numbers to make referencing them easier later on. (note: do not run this command if your images are already labelled 'n.jpg' because it will overwrite some of them

~~~
cd .../JPEGImages
count=1
for file in $PWD/*.jpg
do
mv $file $count.jpg
count=$((count+1))
done
~~~

Important: LabelImg grabs the folder name when writing the xml files and this needs to be VOC2012. We will fix the error that this leads to in the next step.

Run LabelImg. Download a release from https://tzutalin.github.io/labelImg/ then just extract it and run sudo ./labelImg (it segfaults without sudo)

* set autosave on
* set the load and save directories (save should be .../Annotations, load is .../JPEGImages)
* set the default classname to something easy to remember
* press d to move to the next image
* press w to add a box
* Label all examples of the relevant classes in the dataset

From the Annotations dir run
~~~
for file in $PWD/*.xml
do sed -i 's/>JPEGImages</>VOC2012</g' $file
done
Cd to the JPEGImages dir and run the command
~~~
~~~
ls | grep .jpg | sed "s/.jpg//g" > aeroplane_trainval.txt
cp aeroplane_trainval.txt trainval.txt
mv *.txt ../ImageSets/Main/
~~~
The Pascal VOC type dataset should now be all created. If you messed up any of the folder structure, you will need to change the XML file contents. If you rename any of the JPEG files you will need to change both the aeroplane_trainval.txt and XML file contents.

Open bash in models/research and run the following command 'python object_detection/create_pascal_record.py -h' follow the help instructions to create a pascal.record and file from the dataset.


steps:-

step-1:
~~~
genius@Genius:/var/www/html/Reinforcement/CNN/FasterRCNNTutorial-master$ cd models/research
~~~
step-2: 
~~~
genius@Genius:/var/www/html/Reinforcement/CNN/FasterRCNNTutorial-master/models/research$ /home/data/Desktop/final_project/project_ocr/protoc_3.3/bin/protoc object_detection/protos/*.proto --python_out=.
~~~
step-3: 
~~~
genius@Genius:/var/www/html/Reinforcement/CNN/FasterRCNNTutorial-master/models/research$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
~~~
step-4: 
~~~
python object_detection/dataset_tools/create_pascal_tf_record.py -h
~~~
step-5:
~~~
python object_detection/dataset_tools/create_pascal_tf_record.py --data_dir=/home/data/Desktop/final_project/project_ocr/project_segmentation/VOCtemplate --year=VOC2012 --output_path=/home/data/Desktop/final_project/project_ocr/project_segmentation/pascal.record --label_map_path=/home/data/Desktop/final_project/project_ocr/project_segmentation/label.pbtxt --set=trainval
~~~

step-6: 
~~~
python object_detection/legacy/train.py --train_dir=/home/data/Desktop/final_project/project_ocr/project_segmentation/train --pipeline_config_path=/home/data/Desktop/final_project/project_ocr/project_segmentation/faster_rcnn_resnet101_coco.config
~~~
