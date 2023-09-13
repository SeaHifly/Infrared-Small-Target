------------------------------------------------------------------------------
Correction 20230811

There are ridiculous errors in the speed evaluation of our paper. In the past, we choose to evaluate the speed by the following way:

machine warm-up

for:

t1 = time.time()

output = model(img)

t2 = time.time()

whole_time += (t2 - t1)

whole_count += 1

end

avg_FPS=whole_count/whole_time (1)


The speed valuation of MDvsFA by (1) is up to about 300 FPS. Although we used to doubt the results due to its acutual large computations and basic feeling.
However, we choosed to believe the reults veritably and and no reviwers doubt and remind us before the last publication.
Recently, we re-check the speed by the following way:

machine warm-up

t1 = time.time()

for:

output = model(img)

whole_count += 1

end

t2 = time.time()

whole_time += (t2 - t1)

avg_FPS=whole_count/whole_time (2)


Then the speed of MDvsFA drops to only 5.8 FPS for 288×384 image processing, which is much more consistent with the large computation.
After checking other speed values, the FPS values of other methods including ours also drops by some. 
The corerction will be submitted to TGRS recently.
(It doesn't influence most conclusions in the original paper, but some expressions should be re-checked and revised thoughtfully.)

------------------------------------------------------------------------------
Latest infomaition

The datasets will be uploaded to another link. Some information about the datasets will be supplemented.
(https://github.com/SeaHifly/IST-A)

If you find some errors, any corrections (including label failure, code errors and so on) will be welcomed.

email: 
xuhai_0513@foxmail.com. (Choose this first, we will reply within a week.) 
seahifly@hust.edu.cn

--------------------------------------------------------------------------------
Run 'MMRFF_TrainSE_LL01.py' to train your own model on any datasets. 

Run 'MMRFF_TestSE_FL_02.py' to test the model. 

The followed settings should be changed according to your own machine:

(1) 'test_path': the path of images to evaluate. (Only during the testing)

(2) 'model_path': the model path. There is a pretrained model in the default path. (Only during the testing)

(3) 'model_id': the id of the pkl to load. We don't save the best model based on any datasets like some other methods. The model files will be saved once every 30 epochs based on the default settings. We usually take the last model for testing.

(4) 'b_gt_enable': If you are testing some images without GT, you should set it as False. During the training, it must be set as True.

(5) 'train_path': the path of images to train. (Only during the training)

Other settings are feasible and other settings like saving results or showing results will be easy to find.

----------------------------------------------------------------------------------------
Other infomation will be supplemented if needed.

If you find the codes are helpful for your work, please cite the following paper:

H. Xu, S. Zhong, T. Zhang and X. Zou, "Multi-scale Multi-level Residual Feature Fusion for Real-time Infrared Small Target Detection," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2023.3269092.

@ARTICLE{10106465,
  author={Xu, Hai and Zhong, Sheng and Zhang, Tianxu and Zou, Xu},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Multi-scale Multi-level Residual Feature Fusion for Real-time Infrared Small Target Detection}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TGRS.2023.3269092}}
 
----------------------------------------------------------------------------------------
Other notes:

(1) All speed evaluations in the paper are performed without any other speed promotion like onnx or tensorsort. 

(2) This method can only detect targets which are not larger than 16×16 because of some settings. We note that there are some other open-dataset recently. The larger targets may influence the model training. And it's difficult for us to achieve the same performance on IRSTD-1K.

(3) We will train the model on some other open-datasets and present the performance on object-level based on the metrics of correspongding papers as quickly as possible. (We will only try on datasets which consists of images over 10k)
