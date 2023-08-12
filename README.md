# HrSegNet4CrackSegmentation
Real-time High-Resolution Neural Network with Semantic Guidance for Crack Segmentation

# Abs
The current trend in crack detection methods is leaning towards the use of machine learning or deep learning. This is because deep learning-based methods can autonomously extract features from images, thereby avoiding the low stability caused by manually designed operators. However, there are still some problems with the current deep learning-based crack segmentation algorithms. Firstly, the vast majority of research is based on the modification and improvement of commonly used scene segmentation algorithms, with no specifically designed for crack segmentation tasks. Secondly, crack detection is increasingly reliant on edge devices, such as drones and vehicle-mounted cameras. Therefore, the model must be lightweight to achieve real-time segmentation efficiency. However, there is currently limited research in this area. We propose a high-resolution neural network with semantic guidance for real-time crack segmentation, named HrSegNet.

# Update 
2023-08-12
Update [Concrete3k](https://chdeducn-my.sharepoint.com/:u:/g/personal/2018024008_chd_edu_cn/EdzjOhykuQxDjRgs6k-5PU0BtJntPGtTo445f4lBv5HV4Q?e=MCOv5W). In the original Concrete3k, some of the images and labels did not match and we have updated and uploaded them. The results of the corresponding cross-dataset will also be updated.
 
2023-07-17
* Add new datasets: [Asphalt3k](https://chdeducn-my.sharepoint.com/:u:/g/personal/2018024008_chd_edu_cn/EVj4M3fxfcFEuUToiO1QODEBtUuSPXE5FQONgNYti7PDFQ?e=IwZgXT), [Concrete3k](https://chdeducn-my.sharepoint.com/:u:/g/personal/2018024008_chd_edu_cn/Ef-1J7oMk7JHktzA8ildcYQBcqRaz0Er5Y29fN-VQ9SJbw?e=plD8oP). Asphalt3k sourced from [Yang](https://www.mdpi.com/2076-3417/12/19/10089), and Concrete3k sourced from [Wang](https://www.sciencedirect.com/science/article/pii/S0926580522001480).
* Add weight files pre-trained on CrackSeg9k，along with their corresponding training logs.
  
2023-07-02

We are conducting more comparative experiments while using a new pavement dataset that is being manually annotated at the expert level. The results and data will be published soon. We will release the trained model parameters so that you can quickly test them.
### Model Architecture  
![Alt text](./fig/fig1.png)
### [Seg-Grad-CAM](https://arxiv.org/abs/2002.11434)  
![Alt text](./fig/fig5.png)
### Comparisons with state-of-the-art
![Alt text](./fig/fig8.png)


# Data
* [CrackSeg9k](https://github.com/Dhananjay42/crackseg9k) 
* [Asphalt3k](https://chdeducn-my.sharepoint.com/:u:/g/personal/2018024008_chd_edu_cn/EVj4M3fxfcFEuUToiO1QODEBtUuSPXE5FQONgNYti7PDFQ?e=IwZgXT)
* [Concrete3k](https://chdeducn-my.sharepoint.com/:u:/g/personal/2018024008_chd_edu_cn/Ef-1J7oMk7JHktzA8ildcYQBcqRaz0Er5Y29fN-VQ9SJbw?e=plD8oP)

We train the model on a comprehensive dataset (CrackSeg9k) and subsequently transfer to specific downstream scenarios, asphalt (Asphalt3k) and concrete (Concrete3k).
# Installation
The code requires python>=3.8, as well as paddle=2.4.1 and paddleseg=2.7.0 and OpenCV= 4.7.0. You can follow the instructions [paddle](https://github.com/PaddlePaddle/Paddle) and [paddleseg](https://github.com/PaddlePaddle/PaddleSeg) to install all the dependencies. If you need to reproduce the results, you have to install paddle with CUDA support.

# How to use
Once paddle and paddleseg are installed, you can use our published models very easily.  

We start by describing the contents of each directory. The directory `models` defines the high-resolution crack segmentation model we designed, the three model files are almost identical except for the parameter `base`. The model files we are comparing are also included. The directory `configs` is the configuration files for all models, i.e. the details of all training and testing parameters.  

The easiest way to use our models is to use [paddleseg](https://github.com/PaddlePaddle/PaddleSeg). One can put the files of the desired models into the models directory of paddleseg, registering the model using `@manager.MODELS.add_component`. For training the model use the configuration files in the `configs` we provide. 

All data are public available.


# Trained models
**On CrackSeg9k**
* [HrSegNet-B16](https://chdeducn-my.sharepoint.com/:u:/g/personal/2018024008_chd_edu_cn/EZWMNQXFtTpPl-SnUyoKpS0B2EDCDZIn2SX00C0AI_U-Jg?e=o0gqxN)
* [HrSegNet-B32](https://chdeducn-my.sharepoint.com/:u:/g/personal/2018024008_chd_edu_cn/EVaZjUC9tVNMoMkbNOdmemEBh6xPEBUzo2-0ddjGl3bfRQ?e=MWs6Z9)
* [HrSegNet-B48](https://chdeducn-my.sharepoint.com/:u:/g/personal/2018024008_chd_edu_cn/EdoG_do5oFdPmP6NDqWh8AEBh1CfTl6SxD6DX_smxl9WFA?e=WAr0Fi)
* [HrSegNet-B64(bs=16)](https://chdeducn-my.sharepoint.com/:u:/g/personal/2018024008_chd_edu_cn/ETzpUJ9FkN1CoTOO1PB1-68BNYNdqtB0gowlkjzuNJCtQw?e=rCkTGO)
* [HRNet-W18(bs=16)](https://chdeducn-my.sharepoint.com/:u:/g/personal/2018024008_chd_edu_cn/EQcoB7KEbMZHidBi2JchS78BoeI35zALH0m6w3727u7HGA?e=nNDb39)


# Model (TensorRT engine)
We expose all our models as TensorRT, including SOTA for comparison in all experiments. Note that all inputs to the TensorRT engine are **1 × 3 × 400 × 400**. We use TensorRT 8.6.1.
| Model |
| --- |
| [U-Net](https://chdeducn-my.sharepoint.com/:u:/g/personal/2018024008_chd_edu_cn/EYoEi_aQczxOswVyAi8FQBgBYSYXalI8oZKRszWHgbzZwg?e=XuFGzf) |
| [DDRNet](https://chdeducn-my.sharepoint.com/:u:/g/personal/2018024008_chd_edu_cn/EX-QSVExyFVLvasiouuvEwEBe4HPdK3N8HxklK5CAn07DQ?e=DfdBZz) |
| [DeeplabV3+](https://chdeducn-my.sharepoint.com/:u:/g/personal/2018024008_chd_edu_cn/ETkJ1rMqaqBGrfWNg5KCF0EBIxCfYlFk3t0IRD2Uk2cQcA?e=ISPLG0) |
| [OCRNet](https://chdeducn-my.sharepoint.com/:u:/g/personal/2018024008_chd_edu_cn/Ed0l6UAckEFGodrNz1W7aHgBOmoVN6-yZfNIKMTJOp4Fug?e=7u8ZOD) |
| [STDCSeg](https://chdeducn-my.sharepoint.com/:u:/g/personal/2018024008_chd_edu_cn/EV1Rra3XuP5GqImDWMeYdbEBSt64lrmWnAQETKJe0NTO5Q?e=LN0VxD) |
| [BiSeNetV2](https://chdeducn-my.sharepoint.com/:u:/g/personal/2018024008_chd_edu_cn/EfovCQdm_5FJoaySbnd2SBsB2becRV7KTQa7A9_oL7lkHA?e=TI8gZJ) |
| [PSPNet](https://chdeducn-my.sharepoint.com/:u:/g/personal/2018024008_chd_edu_cn/ERTJdaWfJ-9Ess81IwvnBE4Ba0pVnGgyqyZoHFC5hEe1pQ?e=ZzB5Xa) |
| [HrSegNet-B16](https://chdeducn-my.sharepoint.com/:u:/g/personal/2018024008_chd_edu_cn/EYq7OVwYeRtJm0PtXmytSmoB-Ywu8PsC-9eS95V0M7GSpQ?e=1GgLOt) |
| [HrSegNet-B32](https://chdeducn-my.sharepoint.com/:u:/g/personal/2018024008_chd_edu_cn/EURuJVQAW25GnJBvdwW76pgBZdZqyWwT_vifP7Ta98O8_w?e=kKZVLb) |
| [HrSegNet-B48](https://chdeducn-my.sharepoint.com/:u:/g/personal/2018024008_chd_edu_cn/EcUUFXq9dbJHmAz1roiZCMUB3zeM49ILOwzFzHe0iAYS8w?e=SAGci7) |


# Cite
```
@misc{li2023hrsegnet,
      title={HrSegNet : Real-time High-Resolution Neural Network with Semantic Guidance for Crack Segmentation}, 
      author={Yongshang Li and Ronggui Ma and Han Liu and Gaoli Cheng},
      year={2023},
      eprint={2307.00270},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



