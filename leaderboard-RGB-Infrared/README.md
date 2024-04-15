## Cross-Modality RGB-Infrared Person Re-identification LeaderBoard

If you notice any result or the public code that has not been included in this table, please connect [Zhedong Zheng](mailto:zdzheng12@gmail.com) without hesitation to add the method. You are welcomed! 
or create pull request.

Keywords: cross-modality-re-identification, awesome-reid 

 ### Code
 :red_car:  The 1st Place Submission to AICity Challenge 2020 re-id track [[code]](https://github.com/layumi/AICIty-reID-2020)
 [[paper]](https://github.com/layumi/AICIty-reID-2020/blob/master/paper.pdf)
 
 :helicopter:  Drone-based building re-id [[code]](https://github.com/layumi/University1652-Baseline)  [[paper]](https://arxiv.org/abs/2002.12186)

 ### Cross Modality (REGDB Dataset)
|Methods | Rank@1 | mAP| Reference|
| -------- | ----- | ---- | ---- |
|HOG | 13.49% | 10.31% | "[Histograms of oriented gradients for human detection](https://ieeexplore.ieee.org/document/1467360)", Navneet Dalal, Bill Triggs, CVPR 2005|
|LOMO | 0.85% | 2.28% | "[Person Re-identification by Local Maximal Occurrence Representation and Metric Learning](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liao_Person_Re-Identification_by_2015_CVPR_paper.pdf)", Shengcai Liao, Yang Hu, Xiangyu Zhu, Stan Z. Li, CVPR 2015 [**[project]**](http://www.cbsr.ia.ac.cn/users/scliao/projects/lomo_xqda/)|
|MLBP | 2.02% | 6.77% | "[Efficient PSD Constrained Asymmetric Metric Learning for Person Re-Identification](https://ieeexplore.ieee.org/document/1467360)", Shengcai Liao, Stan Z. Li, ICCV 2015 [**[project]**](http://www.cbsr.ia.ac.cn/users/scliao/projects/mlapg/)|
|One-stream Network  | 13.11% | 14.02% | "[RGB-Infrared Cross-Modality Person Re-Identification](https://ieeexplore.ieee.org/document/8237837)", Ancong Wu, Wei-Shi Zheng, Hong-Xing Yu, Shaogang Gong, Jianhuang Lai, ICCV 2017|
|Two-stream Network  | 12.43% | 13.42% | "[RGB-Infrared Cross-Modality Person Re-Identification](https://ieeexplore.ieee.org/document/8237837)", Ancong Wu, Wei-Shi Zheng, Hong-Xing Yu, Shaogang Gong, Jianhuang Lai, ICCV 2017|
|Zero-padding  | 17.75% | 18.90% | "[RGB-Infrared Cross-Modality Person Re-Identification](https://ieeexplore.ieee.org/document/8237837)", Ancong Wu, Wei-Shi Zheng, Hong-Xing Yu, Shaogang Gong, Jianhuang Lai, ICCV 2017|
|TONE  | 16.87% | 14.92% | "[Bi-Directional Center-Constrained Top-Ranking for Visible Thermal Person Re-Identification](https://ieeexplore.ieee.org/document/8732420)", Mang Ye, Xiangyuan Lan, Zheng Wang, Pong C. Yuen, IEEE Transactions on Information Forensics and Security 2020|
|TONE+HCML  | 24.44% | 20.80% | "[Bi-Directional Center-Constrained Top-Ranking for Visible Thermal Person Re-Identification](https://ieeexplore.ieee.org/document/8732420)", Mang Ye, Xiangyuan Lan, Zheng Wang, Pong C. Yuen, IEEE Transactions on Information Forensics and Security 2020|
|BCTR  | 12.43% | 13.42% | "[Visible thermal person re- identification via dual-constrained top-ranking](https://www.ijcai.org/proceedings/2018/0152.pdf)", Mang Ye, Xiangyuan Lan, Zheng Wang, Xiangyuan Lan, Pong C. Yuen, IJCAI 2018|
|BDTR  | 33.47% | 31.83% | "[Visible thermal person re- identification via dual-constrained top-ranking](https://www.ijcai.org/proceedings/2018/0152.pdf)", Mang Ye, Xiangyuan Lan, Zheng Wang, Xiangyuan Lan, Pong C. Yuen, IJCAI 2018|
|HSME  | 41.34% | 38.82% | "[HSME: Hypersphere manifold embedding for visible thermal person re-identification](https://ojs.aaai.org/index.php/AAAI/article/view/4853)", Yi Hao, Nannan Wang, Jie Li, Xinbo Gao, AAAI 2019|
|D2RL  | 43.4% | 44.1% | "[Learning to reduce dual-level discrepancy for infrared-visible person re-identification](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Learning_to_Reduce_Dual-Level_Discrepancy_for_Infrared-Visible_Person_Re-Identification_CVPR_2019_paper.pdf)", Zhixiang Wang, Zheng Wang, Yinqiang Zheng, Yung-Yu Chuang，Shin’ichi Satoh, CVPR 2019|
|MSR  | 48.43% | 48.67% | "[Learning Modality-Specific Representations for Visible-Infrared Person Re-Identification](https://ieeexplore.ieee.org/abstract/document/8765608)", Zhanxiang Feng, Jianhuang Lai, Xiaohua Xie, TIP 2019|
|JSIA  | 48.5% | 49.34% | "[Cross-modality paired-images generation for rgb-infrared person re- identification](https://arxiv.org/abs/2002.04114)", Guan-An Wang, Tianzhu Zhang, Yang Yang, Jian Cheng, Jianlong Chang, Xu Liang, Zengguang Hou, AAAI 2020 [**[code]**](https://github.com/wangguanan/JSIA-ReID)|
|D-HSME  | 50.85% | 47.00% | "[HSME: Hypersphere manifold embedding for visible thermal person re-identification](https://ojs.aaai.org/index.php/AAAI/article/view/4853)", Yi Hao, Nannan Wang, Jie Li, Xinbo Gao, AAAI 2019|
|PIGA  | 52.1% | 51.9% | "[Cross-modality paired-images generation and augmentation for RGB-infrared person re-identification](https://www.sciencedirect.com/science/article/abs/pii/S0893608020301702)", Guan'an Wang, Yang Yang, Tianzhu Zhang, jian Cheng, Zengguang Hou, Prayag Tiwari, Hari Mohan Pandey, Neural Networks 2020 [**[code]**](https://github.com/wangguanan/JSIA-ReID)|
|ABP  | 56.35% | 48.58% | "[Abp: Adaptive body partition model for visible infrared person re-identification](https://ieeexplore.ieee.org/document/9102974)", Ziyu Wei, Xi Yang, Nannan Wang, Xinbo Gao, ICME 2020|
|AlignGAN  | 57.90% | 53.60% | "[Rgb- infrared cross-modality person re-identification via joint pixel and fea- ture alignment](https://arxiv.org/abs/1910.05839)", Guan'an Wang, Tianzhu Zhang, Jian Cheng, Si Liu, Yang Yang, Zengguang Hou, ICCV 2019 [**[code]**](https://github.com/wangguanan/AlignGAN)|
|DAPR  | 61.5% | 59.4% | "[Beyond modality alignment: Learning part-level representation for visible-infrared person re-identification](https://www.sciencedirect.com/science/article/abs/pii/S0262885621000238)", Peng Zhang, Qiang Wu, Xunxiang Yao, Jingsong Xu, Image and Vision Computing 2021|
|XIV  | 62.21% | 60.18% | "[Infrared-visible cross-modal person re-identification with an X modality](https://ojs.aaai.org/index.php/AAAI/article/view/5891)", Diangang Li, Xing Wei, Xiaopeng Hong, Yihong Gong, AAAI 2020|
|mtGAN-D  | 65.6% | 60.0% | "[Modality-transfer generative adversarial network and dual-level unified latent representation for visible thermal person re-identification](https://link.springer.com/article/10.1007/s00371-020-02015-z)", Xing Fan, Wei Jiang, Hao Luo, Weijie Mao, Visual Computer 2020|
|DDAG  | 69.34% | 63.46% | "[Dynamic dual-attentive aggregation learning for visible-infrared person re- identification](https://arxiv.org/abs/2007.09314)", Mang Ye, Jianbing Shen, David J. Crandall, Ling Shao, Jiebo Luo, arXiv:2007.09314 [**[code]**](https://github.com/mangye16/DDAG)|
|Hi-CMD  | 70.93% | 66.04% | "[Hi-CMD: Hierarchical Cross-Modality Disentanglement for Visible-Infrared Person Re-Identification](https://arxiv.org/abs/1912.01230)", Seokeon Choi, Sumin Lee, Youngeun Kim, Taekyung Kim, Changick Kim, CVPR 2020|
|HAT  | 71.83% | 67.56% | "[Visible-Infrared Person Re-Identification via Homogeneous Augmented Tri-Modal Learning](https://ieeexplore.ieee.org/document/9115075)", Mang Ye, Jianbing Shen, Ling Shao, TIFS 2020|
|MACE   | 72.37% | 69.09% | "[Cross-modality person re-identification via modality-aware collaborative ensemble learning](https://ieeexplore.ieee.org/document/9107428)", Mang Ye, Xiangyuan Lan, Qingming Leng, and Jianbing Shen, TIP 2020|
|ADCNet   | 72.9% | 66.5% | "[Adversarial disentanglement and corre- lation network for rgb-infrared person re-identification](https://ieeexplore.ieee.org/document/9428376)", Bingyu Hu, Jiawei Liu, Zheng-jun Zha, ICME 2021|
|DG-VAE  | 72.97% | 71.78% | "[Dual gaussian-based variational subspace disentanglement for visible-infrared person re-identification](https://arxiv.org/abs/2008.02520)", Nan Pu, Wei Chen, Yu Liu, Erwin M. Bakker, Michael S. Lew, ACM MM 2020|
|FBP-AL  | 73.98% | 68.24% | "[Flexible body partition-based adversarial learning for visible infrared person re-identification](https://ieeexplore.ieee.org/document/9367015)", Ziyu Wei, Xi Yang, Nannan Wang, Xinbo Gao, IEEE Transactions on Neural Networks and Learning Systems 2021|
|LbA  | 74.17% | 67.64% | "[Learning by aligning: Visible- infrared person re-identification using cross-modal correspondences](https://arxiv.org/abs/2108.07422)", Hyunjong Park, Sanghoon Lee, Junghyup Lee, Bumsub Ham, ICCV 2021 [**[project]**](https://cvlab.yonsei.ac.kr/projects/LbA/)|
|CICL  | 78.8% | 69.4% | "[Joint Color-irrelevant Consistency Learning and Identity-aware Modality Adaptation for Visible-infrared Cross Modality Person Re-identification](https://ojs.aaai.org/index.php/AAAI/article/view/16466)", Zhiwei Zhao, Bin Liu, Qi Chu, Yan Lu, Nenghai Yu, AAAI 2021|
|NFS  | 80.54% | 72.10% | "[Neural feature search for rgb-infrared person re-identification](https://arxiv.org/abs/2104.02366)", Yehansen Chen, Lin Wan, Zhihang Li, Qianyan Jing, Zongyuan Sun, CVPR 2021|
|DGTL  | 83.92% | 73.78% | "[Strong but simple baseline with dual-granularity triplet loss for visible-thermal person re-identification](https://ieeexplore.ieee.org/document/9376983)", Haijun Liu, Yanxia Chai, Xiaoheng Tan, Dong Li and Xichuan Zhou, IEEE Signal Processing Letters 2021 [**[code]**](https://github.com/hijune6/DGTL-for-VT-ReID)|
|WIT  | 85.0% | 75.9% | "[Visible-infrared cross-modality person re-identification based on whole-individual training](https://www.sciencedirect.com/science/article/abs/pii/S0925231221001491?via%3Dihub)", Jia Sun, Yanfeng Li, Houjin Chen, Yahui Peng, Xiaodi Zhu, Neurocomputing 2021|
|HCT  | 91.05% | 83.28% | "[Parameter Sharing Exploration and Hetero-center Triplet Loss for Visible-Thermal Person Re-Identification](https://arxiv.org/abs/2008.06223)", Haijun Liu, Xiaoheng Tan, Xichuan Zhou, TMM 2020 [**[code]**](https://github.com/hijune6/Hetero-center-triplet-loss-for-VT-Re-ID)|
|GLMC  | 91.84% | 81.42% | "[Global-Local Multiple Granularity Learning for Cross-Modality Visible-Infrared Person Reidentification](https://ieeexplore.ieee.org/document/9457243)", Jia Sun, Yanfeng Li, Houjin Chen, Yahui Peng, Xiaodi Zhu, TNNLS 2021|
|HHRG  | 94.92% | 94.58% | "[Homogeneous and Heterogeneous Relational Graph for Visible-infrared Person Re-identification](https://arxiv.org/abs/2109.08811)", Yujian Feng, Feng Chen, Jian Yu, Yimu Ji, Shangdong Liu [**[code]**](https://github.com/fegnyujian/Homogeneous-and-Heterogeneous-Relational-Graph)|
