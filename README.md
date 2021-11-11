# Infrared-Small-Target
Just share some data


Here, we shared a infrared small target dataset and we wish it could be valuable for researchers all over the world.
It may involve in some military application, but we wish it can be beneficial to some civilian use such as UAV or bird alarming in airports, birds migration monitoring or something else.

Images in training dataset or test dataset are picked out from 72 image sequences in a sequencial traversal style. The total number of orginal images is over 22K, but only 10117 training images and 2476 test image are picked out. For each sequence, only when the index of current traversal image is 25 larger than the index of last valid or the SSIM between current traversal image and last valid image is smaller than 0.99 will be current traversal image treated as a valid image. The valid images will be randomly distributed to training dataset or test dataset. Of course, the first image in each sequence will be treated as a valid imag so the traversal processing can be implemented.

All images share a same size of 288×384 and the width or height of all targets are smaller than 16. We take birds, UAVs, helicopters, planes and aircrafts which can fly in a relatively far distance as infrared small targets.

It's not proper for any researcher to share the codes, but it's workable for everyone to share their result in a common and same dataset.



Any question will be welcomed in haixu_hifly@qq.com.
