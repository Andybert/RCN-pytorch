# RCN-pytorch
This is an implementation of RCN (https://github.com/niallmcl/Recurrent-Convolutional-Video-ReID) based on Pytorch (1.7.0) framework

Please prepare the optical flow data first. Then train and test by:
python train_test.py --dir_RGB RGBdataDir --dir_OF opticalFlowDataDir

For other setting, e.g. GPU, margin for contrastive (HingeEmbeddingLoss) loss, the symbol definition can be seen in the option.py file.
