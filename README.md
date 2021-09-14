# DANN
DANN uses gradient reversal to align the source and the target distribution globally, that are obtained from the deep features of the Convolutional Network

# CutMix
cutMix combines two images by retaining features of the source image and the target image in a random ratio.

# cutMix-DANN
Combining CutMix with traditional DANN to improve upon the target dataset accuracy for a domain adaptation task. The encoder is jointly trained on the source and cutMix data

![image](https://user-images.githubusercontent.com/32479901/129860759-68c047a9-b703-43d7-8cea-980ba78001ab.png)



References - 
1. Yun, Sangdoo et al. “CutMix: Regularization Strategy to Train Strong Classifiers With Localizable Features.” 2019 IEEE/CVF International Conference on Computer Vision (ICCV) (2019): 6022-6031.
2. Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, and V. Lempitsky, “Domain-adversarial training of
neural networks,” J. Mach. Learn. Res., vol. 17, pp. 59:1–59:35, 2016
