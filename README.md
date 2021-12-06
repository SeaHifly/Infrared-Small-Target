# Infrared-Small-Target-A
Here, we shared a infrared small target dataset and we wish it could be valuable for researchers all over the world.
It may involve in some military application, but we wish it can be beneficial to some civilian use such as UAV or bird alarming in airports, birds migration monitoring or something else. We name this dataset as 'infrared small target-A ' dateset, simplified as 'IST-A' dataset.

Images in training dataset or test dataset are picked out from 72 image sequences in a sequencial traversal style. The total number of orginal images is over 22K, but only 10117 training images and 2476 test image are picked out. For each sequence, only when the index of current traversal image is 25 larger than the index of last valid image or the SSIM between current traversal image and last valid image is smaller than 0.99 will be current traversal image treated as a valid image. The valid images will be randomly distributed to training dataset or test dataset. Of course, the first image in each sequence will be treated as a valid image so the traversal processing can be implemented.

All images share a same size of 288×384 and the width or height of all targets are smaller than 16. We take birds, UAVs, helicopters, planes and aircrafts which can fly in a relatively far distance as infrared small targets.

Each image in training dataset or  test dataset is paired with a same name txt file. For example, there is an image named "000003.png" in test dataset, then its paired txt file is "000003.txt" and the ground truth of target location is written in such txt files. 

It's not proper to demand all researchers to share the codes, but it's workable for everyone to share their result in a common and same dataset.
More details about the dataset will be supplemented. 

The dataset was first introduced in XXXXXXXXXXXXXX, if you think it's beneficial to you, please consider citing  XXXXXXXXXXX.
Before we complete last paragraph, if you want to use our dataset in some papers, please contact us firstly.

Some methods cited in XXXXXXX are not the official codes, so some evaluation values may be different from some papers. Anyone who possess the official codes and would like to share the results with us, please contact us. Any contribution will be welcomed, and we'd love to add your name in our paper before the last acceptance if we get your permission. 


Any question and contributions will be welcomed!
email : seahifly@hust.edu.cn
