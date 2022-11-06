# Infrared-Small-Target-A
=====================================================================================================================
Latest Information




=====================================================================================================================
Notes edited few month ago.

I am sorry to remove all previous data because I don't want to receive too many letters inquiring about my experiment results or some other else. (20220904)
Please wait and the release is coming.

Extended dataset with over 60K (68311) images will be released after the acceptance of our paper. 
We will try our best to make it and release the dataset and codes as quickly as possible.
(2022 08 14)
The original dataset has been removed from here. We will upload the extended dataset directly.

Here, we shared an infrared small target dataset and we wish it could be valuable for researchers all over the world.
It may involve some military application, but we wish it can be beneficial to some civilian use such as UAV or bird alarming in airports, bird migration monitoring, or something else. We name this dataset the 'infrared small target-A ' dataset, simplified as the 'IST-A' dataset.

The dataset IST-B mentioned in our paper can't be shared due to some special reasons.
We are collecting more images with infrared small targets in the sky and more images with box labels may be released.






Images in the training dataset or test dataset are picked out from 72 image sequences in a sequential traversal style. The total number of original images is over 22K, but only 10117 training images and 2476 test images are picked out. For each sequence, only when the index of the current traversal image is 25 larger than the index of the last valid image or the SSIM between the current traversal image and the last valid image is smaller than 0.99 will be current traversal image treated as a valid image. The valid images will be randomly distributed to the training dataset or test dataset. Of course, the first image in each sequence will be treated as a valid image so the traversal processing can be implemented.

All images share the same size of 288×384 and the width or height of all targets are smaller than 16. We take birds, UAVs, helicopters, planes, and aircraft which can fly at a relatively far distance as infrared small targets.

Each image in the training dataset or test dataset is paired with the same name txt file. For example, if there is an image named "000003.png" in the test dataset, then its paired txt file is "000003.txt" and the ground truth of the target location is written in such txt files. 

It's not proper to demand all researchers to share the codes, but it's workable for everyone to share their results in a common and same dataset.
More details about the dataset will be supplemented. 

=======================================================================================================================
The dataset was first introduced in XXXXXXXXXXXXXX, Some methods cited in this paper are not the official codes, so some evaluation values may be different from some papers. Anyone who possesses the official codes and would like to share the results with us, please contact us. Any contribution will be welcomed, and we'd love to add your name to our paper before the last acceptance if we get your permission. 

Before we complete the last paragraph, if you want to use our dataset in some papers, please contact us firstly.
If you think it's beneficial to you, please consider citing  XXXXXXXXXXX.

Any question and contributions will be welcomed!
email : seahifly@hust.edu.cn or xuhai_0513@foxmail.com
