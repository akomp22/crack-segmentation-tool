# Crack segmentation tool

The presented crack segmentation tool is designed to generate pixel-wise labaling of images of cracks in a semi-automatic manner. Its main goal is to simplify creation of a dataset to train deep learning algorithm for crack segmentation that often done fully manually. The mathematical underpin for the algorithm was developed by Remco Duits(https://www.win.tue.nl/~rduits/) and group of mathematical image analysis of Eindhoven University of Technology.

The segmentation process consists of few main steps:
1. Manual selection of two crack end-points
2. Finding of a crack path between the selected points
3. Crack edges along the retrieved path are found
4. Also crack segment can be added by manually drawing crack contur
5. Pixels between crack edges are marked as crack pixels.

The description of the algorithm can be found in [1] (conference paper to be added)

The tool can be used in two ways. 
 - As an app with a graphical user interface (see tuturial bellow)
 - As python functions (see file Examples.ipynb)

The app designed to segment images from a specific folder. After segmentation JSON files for each image will be created. "check annotetions.ipynb" shows how to read annotation JSON files.

## App tuturial

#### Start the tool
1. Clone the repository or download archive and unzip it on your local machine. Run the app.py python script

![](https://github.com/akomp22/crack-segmentation-tool/blob/main/gif/1.gif)


#### Start segmentation
1. Enter path to a folder that contain images you want to segment. 
2. Press "Start"
3. If at anny time during segmentation you decide to skip the image press "Skip" button. The empty annotation file will be saved.



![](https://github.com/akomp22/crack-segmentation-tool/blob/main/gif/2.gif)



#### Select crack endpoints
1. Press "Select Crack End-Points" to open window for point selection. 
2. Adjust Adjust "Image size" to fit the selection window fit in your screen. 
3. Use the mouse wheel to zoom in/out around current cursor position (to drag image use sequentially zoom in/out and move cursor).
4. Press left mouse button to select point.
5. Press right mouse button to undo previouse point selection
6. Choose two points on image (crack tips)
7. Press ESC to end selection

![](https://github.com/akomp22/crack-segmentation-tool/blob/main/gif/3.gif)

#### Crop image around crack
1. Choose "Dark crack" if crack is darker then background and vice versa
2. Only one color chennel should be chosen for processing. Choose the one that gives the most of the the contrast
3. Downsampling reduces image size to reduce processing time. To course image also reduces tracking accuracy.
4. X and Y margins determine offset of cropped image from the selected points. Choose it to fit crack into cropped image and not to make image size to big
5. Indicetion of the coped image size below the "Update croped image" button helps you to estimate processing time (having some experience with it)
![](https://github.com/akomp22/crack-segmentation-tool/blob/main/gif/4.gif)

#### (Optional) Design wavelet to create orientation scores (modified version of the croped image) and compare with crack
1. Descriptio of paramters and they influence on the cake wavelet can be found in [2](reference to be added)
2. Press "check cake wavelet" to display the wavelet
3. Press "select crack point to check width" to select middle point of a crack with representative crack width. Press "Update" to display crack middle-point
4. Compare cake wavelet and crack middlepoint. You want to design wavalet so that the width of its bright middle region close to width of a crack
![](https://github.com/akomp22/crack-segmentation-tool/blob/main/gif/5.gif)

#### Creat process and check orientation scores
1. Press "update OS"
2. Press "Update cost"
3. Adjust cost parameters to get the best crack response on the projection displayed
![](https://github.com/akomp22/crack-segmentation-tool/blob/main/gif/6.gif)

#### Crack track
1. Press "crack track" to run the crack tracking algorithm
2. Description of g11, g22 and g33 parameters are given in [1]
3. "Track width" and "Track color" regulates only visualisation of the track
4. Press "Update track display"
5. If you want to check the retrieved crack track you can use "Track full screen" button
Note: if crack edges have good visibility, the crack track can be made rough. You can always refine the crack track by decreasing image crop downsample factor or other parameters
![](https://github.com/akomp22/crack-segmentation-tool/blob/main/gif/7.gif)

#### Crack edges retrieving
1. Go to Segmentation Tab
2. Select filter size. The filter size defines how far from the retrieved crack track crack edges can be found  
3. Press "Edge mask"
4. Adjust "mu","l" and "p" parameters to get the best results
5. Press "Edge tracks" button
6. To display the retrived crack conturs in ful screen mode press "Edge tracks full screen"
![](https://github.com/akomp22/crack-segmentation-tool/blob/main/gif/8.gif)

#### Save segment and annotation
You can repate this steps multiple times until you get desired results. After thet you can press "save segment" button to check created segmentation mask. After you saved a segment you can create another segment on the same image by repeating the described procedure. Alternativaly you can go to "Add segment manually" tab, and save a segment created in this tab. After all crack segments are added to segmentation mask, you can go to the "Save annotation tab" to check and save annotations. After pressing the "Save annotation" button, a file with the same name as the processed image and with JSON extension is created. This file contain list of pixels postions that was inside saved segments. On instruction on how to read this file in your python code refer to "check annotation.ipynb" file of this repository.

#### Add segment manually.
1. You can also create a crack segment manually by drawing crack conturs. For this go to "Add segment manually" tab and press "Draw segment"
2. Use mouse wheel for zoom in/out
3. Crack contur consists of parts. Each time you press left mouse button (LMB), new part of contur is started. Hold left mouse button and move cursor to draw a contur part. To fix the drawn part you can release the LMB. If you press LMB again it will start new part of contur, connecting end of previous part with the beginning of new part by a straight line. 
4. Press right mouse button to erase previous contur part.
5. Press Esc to exit the drawing window. End of the last contur part and beggining of the first contur part will be connected byt a straight line to close a contur.
![](https://github.com/akomp22/crack-segmentation-tool/blob/main/gif/9.gif)




