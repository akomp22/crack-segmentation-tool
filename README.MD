# Crack annotation tool

The presented crack annotation tool is designed to generate pixel-wise labeling of images of cracks in a semi-automatic manner. It also allows to draw bounding boxes around objects. Its main goal is to simplify the creation of a dataset to train deep learning algorithms for crack segmentation that is often done manually. The mathematical underpinning for the segmentation algorithm was developed by Remco Duits(https://www.win.tue.nl/~rduits/) and a group of mathematical image analysis at Eindhoven University of Technology.

The annotation can be made in one of two ways or both:
1. Bounding box annotation
2. Crack Segmentation

The segmentation process consists of a few main steps:
1. Manual selection of two crack end-points
2. Finding a crack path between the selected points
3. Finding of crack edges along the retrieved path
4. Pixels between crack edges are marked as crack pixels.
5. Optionally, the crack segment can be added by manually drawing the crack contour


The description of the algorithm can be found in our paper:

[1] Kompanets, A., Duits, R., Leonetti, D., van den Berg, N., Snijder, H.H. (2023). Segmentation Tool for Images of Cracks. In: Skatulla, S., Beushausen, H. (eds) Advances in Information Technology in Civil and Building Engineering. ICCCBE 2022. Lecture Notes in Civil Engineering, vol 357. Springer, Cham. https://doi.org/10.1007/978-3-031-35399-4_8

Preprint also available at: https://www.win.tue.nl/~rduits/CD.pdf 

The tool can be used in two ways. 
 - As an app with a graphical user interface (see tutorial bellow)
 - As python functions (see file Examples.ipynb)

The app is designed to create annotation JSON files for images from a specified folder. "generate_mask_from_json.ipynb" file in this repository shows how to read annotations from the JSON files.
If parameters are set properly, it should take less than a minute to segment a single crack on an image. If it takes longer, consider increasing the downsampling factor (see explanations below).
## Specific dependencies
- python 3.9.15 (Not tested on later versions)
- agd
- hfm 0.2.13 (https://github.com/Mirebeau/HamiltonFastMarching)

to install libraries:
```console
conda install agd
conda install -c agd-lbr hfm=0.2.13
```
## Cite
If you use this code or the app in your own work please cite our paper:

[1] Kompanets, A., Duits, R., Leonetti, D., van den Berg, N., Snijder, H.H. (2023). Segmentation Tool for Images of Cracks. In: Skatulla, S., Beushausen, H. (eds) Advances in Information Technology in Civil and Building Engineering. ICCCBE 2022. Lecture Notes in Civil Engineering, vol 357. Springer, Cham. https://doi.org/10.1007/978-3-031-35399-4_8

# App tutorial

#### Run the tool
1. Clone the repository or download the archive and unzip it on your local machine. Run the crack_tool.py python script

## "Select Image" Tab
#### Start annotation
1. Enter the path to a folder that contains the images you want to segment. 
2. Press "Select Folder"

You will see a selectable list of images in the folder. If an annotation file for this image exists, the annotation will be displayed.

![](https://github.com/akomp22/crack-segmentation-tool/blob/main/video/gif/1.gif)

#### Annotate bounding box (Optional)
1. Select "Image size" parameter to adjust the size of a window for drawing
2. Press the "Draw Box" button to open the window for drawing bounding boxes
3. Draw a single bounding box
    - Scroll mouse will Zoom in/out image
    - Click the left mouse button to select the first corner of a bounding box
    - Click the left mouse button once again to select the oposite corner of the bounding box
    - Press the right mouse button to undo the previous selection
    - Press Esc to escape the drawing window
3. Select a positive integer in the "Class" window number to indicate the calls of the bounding box that was just drawn. (Note: in the app, the class 0 is shown to be for crack and class 1 for corrosion, however, you can choose any integer number for any annotated object)
4. Press the "save Box" button to save the coordinates and class of the bounding box to the annotation file


Now you can select another image to continue annotating the bounding box or go to the "Tracking" Tab to annotate the crack segmentation 

5. To delete the bounding box from the annotation file press the "Clear Boxes" button
6. To delete annotated segmentations from the annotation file press the "Clear segmentation" button

![](https://github.com/akomp22/crack-segmentation-tool/blob/main/video/gif/2.gif)


## "Tracking" Tab
#### Select crack endpoints
1. Press "Select Crack End-Points" to open the window for point selection. 
2. Adjust "Image size" to fit the selection window to your screen. 
3. Use the mouse wheel to zoom in/out around the current cursor position (to drag an image use sequentially zoom in/out and move the cursor).
4. Press the left mouse button to select the point.
5. Press the right mouse button to undo previous point selection
6. Choose two points on the image (crack tips)
7. Press ESC to end selection

#### Crop image around crack
1. Choose "Dark crack" if the crack is darker than the background and vice versa
2. Only one color channel should be chosen for processing. Choose the one that gives the most of the contrast
3. Downsampling reduces the image size to reduce processing time. Too course image also reduces tracking accuracy.
4. X and Y margins determine the offset of the cropped image from the selected points. Choose it to fit crack into the cropped image and not to make the image size too big
5. Indicetion of the coped image size below the "Update cropped image" button helps you to estimate processing time (having some experience with it)

#### (Optional) Design wavelet to create orientation scores (modified version of the cropped image) and compare with crack
1. A description of parameters and their influence on the cake wavelet can be found in [2](a reference to be added)
2. Press "check cake wavelet" to display the wavelet
3. Press "select the crack point to check width" to select a middle point of a crack with a representative crack width. Press "Update" to display cracked middle-point
4. Compare cake wavelet and crack middlepoint. You want to design the wavelet so that the width of its bright middle region close to the width of a crack

#### Create process and check orientation scores
1. Press "update OS"
2. Press "Update cost"
3. Adjust cost parameters to get the best crack response on the projection displayed

#### Crack track
1. Press "crack track" to run the crack tracking algorithm
2. Description of g11, g22 and g33 parameters are given in [1]
3. "Track width" and "Track color" regulates only visualization of the track
4. Press "Update track display"
5. If you want to check the retrieved crack track you can use the "Track full screen" button
Note: if crack edges have good visibility, the crack track can be made rough. You can always refine the crack track by decreasing the image crop downsample factor or other parameters

![](https://github.com/akomp22/crack-segmentation-tool/blob/main/video/gif/3.gif)

## "Segmentation" Tab

#### Crack edges retrieving
1. Go to the Segmentation Tab
2. Select filter size. The filter size defines how far from the retrieved crack track crack edges can be found  
3. Press "Edge mask"
4. Adjust the "mu", "l" and "p" parameters to get the best results
5. Press the "Edge tracks" button
6. To display the retrieved crack contours in full-screen mode press "Edge tracks full screen"


#### Save segment
You can repeat these steps multiple times until you get the desired results. After that, you can press the "save segment" button to check the created segmentation mask. After you save a segment you can create another segment on the same image by repeating the described procedure. Alternatively, you can go to the "Add segment manually" tab, and save a segment created in this tab. After all crack segments are added to the segmentation mask, you can go to the "Save annotation tab" to check and save annotations. After pressing the "Save annotation" button, a file with the same name as the processed image and with JSON extension is created. This file contains a list of pixel positions that were inside saved segments. For instructions on how to read this file in your python code refer to the "check annotation.ipynb" file of this repository.

![](https://github.com/akomp22/crack-segmentation-tool/blob/main/video/gif/4.gif)

## "Add segment manually" Tab
#### Add segment manually.
1. You can also create a crack segment manually by drawing crack contours. For this go to the "Add segment manually" tab and press "Draw segment"
2. Use the mouse wheel to zoom in/out
3. Crack contur consists of parts. Each time you press the left mouse button (LMB), part of the contour is started. Hold the left mouse button and move the cursor to draw a contour part. To fix the drawn part you can release the LMB. If you press LMB again it will start a new part of the contur, connecting the end of the previous part with the beginning of the new part by a straight line. 
4. Press the right mouse button to erase the previous contour part.
5. Press Esc to exit the drawing window. The end of the last contour part and the beginning of the first contour part will be connected by a straight line to close a contour.

![](https://github.com/akomp22/crack-segmentation-tool/blob/main/video/gif/5.gif)

![](https://github.com/akomp22/crack-segmentation-tool/blob/main/screen-capture.webm)
