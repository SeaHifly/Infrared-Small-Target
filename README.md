------------------------------------------------------------------------------
Latest infomaition

The datasets will be uploaded to another link. Some information about the datasets will be supplemented.
(https://github.com/SeaHifly/IST-A)

If you want to take the dataset for your own research or paper, please contact us. We will upload a version to the arxiv as quickly as possible so that one can cite it.

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

Other settings are feasible. For example, you can increase the channels setting or change the batch size based on your own machine.

----------------------------------------------------------------------------------------
Other infomation will be supplemented if needed.




