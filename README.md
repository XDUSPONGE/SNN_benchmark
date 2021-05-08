# Spiking Neural Network Paper List
## Framework
* BindsNet[[code](https://github.com/BindsNET/bindsnet)] <br />
* Brain2[[code](https://github.com/brian-team/brian2)] <br />
* SpykeTorch[[code](https://github.com/miladmozafari/SpykeTorch)] <br />
* Norse[[code](https://github.com/norse/norse)] <br />
* SpikingJelly[[code](https://github.com/fangwei123456/SpikingFlow)] <br />
* Nengo[[code](https://github.com/nengo/nengo)] <br />
* PySNN[[code](https://github.com/BasBuller/PySNN)] <br />
* SNN_toolbox[[code](https://github.com/NeuromorphicProcessorProject/snn_toolbox)] <br />
## SNN Adversarial Robustness
* Saima Sharmin, Nitin Rathi, Priyadarshini Panda, Kaushik Roy ***ECCV 2020***<br />
" Inherent Adversarial Robustness of Deep Spiking Neural Networks: Effects of Discrete Input Encoding and Non-Linear Activations"
[[paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/html/6748_ECCV_2020_paper.php)]
 [[code](https://github.com/ssharmin/spikingNN-adversarial-attack)]
* Saima Sharmin, Priyadarshini Panda, Syed Shakib Sarwar, Chankyu Lee, Wachirawit Ponghiran, Kaushik Roy ***IJCNN 2019*** <br />
"A Comprehensive Analysis on Adversarial Robustness of Spiking Neural Networks".
 [[paper](https://ieeexplore.ieee.org/document/8851732)]
* Akhilesh Jaiswal, Amogh Agrawal, Indranil Chakraborty, Deboleena Roy, Kaushik Roy ***IJCNN 2019***<br />
"On Robustness of Spin-Orbit-Torque Based Stochastic Sigmoid Neurons for Spiking Neural Networks".
 [[paper](https://ieeexplore.ieee.org/document/8851780)]
 * Xueyuan She, Yun Long, Saibal Mukhopadhyay ***IJCNN 2019***<br />
"Improving Robustness of ReRAM-based Spiking Neural Network Accelerator with Stochastic Spike-timing-dependent-plasticity".
 [[paper](https://ieeexplore.ieee.org/document/8851825)]
## Other Application
* Allan Mancoo, Sander W. Keemink, Christian K. Machens  ***NIPS 2020***<br />
"Understanding spiking networks through convex optimization"
 [[paper](https://proceedings.neurips.cc/paper/2020/file/64714a86909d401f8feb83e8c2d94b23-Paper.pdf)]
 * Seijoon Kim, Seongsik Park, Byunggook Na, Sungroh Yoon ***AAAI 2020***<br />
**Spiking-YOLO:** "Spiking Neural Network for Energy-Efficient Object Detection"
 [[paper](https://arxiv.org/pdf/1903.06530.pdf)]
 * Biswadeep Chakraborty, Xueyuan She <br />
"A Fully Spiking Hybrid Neural Network for Energy-Efficient Object Detection"
 [[paper](https://arxiv.org/abs/2104.10719)]
## Papers
For the Spiking Neural Network studies, it can be roughly divided into three categories
* The Conversion Method (Converting a well-trained ann to snn)
* SNN trained with BP
* SNN trained with Biological Plasticity Rules (STDP, Hebbian,etc)
### Conversion Based Methods
* Weihao Tan, Devdhar Patel, Robert Kozma ***AAAI 2021***<br />
"Strategy and Benchmark for Converting Deep Q-Networks to Event-Driven Spiking Neural Networks"
 [[paper](https://arxiv.org/pdf/2009.14456.pdf)]
* Zhanglu Yan, Jun Zhou, Weng-Fai Wong ***AAAI 2021***<br />
"Near Lossless Transfer Learning for Spiking Neural Networks"
 [[paper](https://www.comp.nus.edu.sg/~wongwf/papers/AAAI-2021.pdf)]
* Bing Han, Gopalakrishnan Srinivasan, and Kaushik Roy ***CVPR 2020***<br />
"RMP-SNN: Residual Membrane Potential Neuron for Enabling Deeper High-Accuracy and Low-Latency Spiking Neural Network"
[[paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Han_RMP-SNN_Residual_Membrane_Potential_Neuron_for_Enabling_Deeper_High-Accuracy_and_CVPR_2020_paper.html)]
* Bing Han, Kaushik Roy ***ECCV 2020*** <br />
"Deep Spiking Neural Network: Energy Efficiency Through Time based Coding"
[[paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1103_ECCV_2020_paper.php)]
 * Nitin Rathi, Gopalakrishnan Srinivasan, Priyadarshini Panda, Kaushik Roy ***ICLR 2020***<br />
"Enabling Deep Spiking Neural Networks with Hybrid Conversion and Spike Timing Dependent Backpropagation".
 [[paper](https://arxiv.org/pdf/2005.01807.pdf)]
 [[code](https://github.com/nitin-rathi/hybrid-snn-conversion.git)]
 * Lei Zhang, Shengyuan Zhou, Tian Zhi, Zidong Du, Yunji Chen ***AAAI 2019***<br />
**TDSNN** "DFrom Deep Neural Networks to Deep Spike Neural Networks with Temporal-Coding".
 [[paper](https://aaai.org/ojs/index.php/AAAI/article/view/3931)]
 * Ruizhi Chen, Hong Ma, Shaolin Xie, Peng Guo, Pin Li, Donglin Wang ***IJCNN 2018***<br />
"Fast and Efficient Deep Sparse Multi-Strength Spiking Neural Networks with Dynamic Pruning".
  * Jingling Li, Weitai Hu, Ye Yuan, Hong Huo, Tao Fang ***ICONIP 2017***<br />
"Bio-Inspired Deep Spiking Neural Network for Image Classification".
 [[paper](https://link.springer.com/chapter/10.1007%2F978-3-319-70096-0_31)]
 [[paper](https://ieeexplore.ieee.org/document/8489339)]

### SNN trained with BP
* Shibo Zhou, Xiaohua LI, Ying Chen, Sanjeev T. Chandrasekaran, Arindam Sanyal ***AAAI 2021***<br />
"Temporal-Coded Deep Spiking Neural Network with Easy Training and Robust Performance"
 [[paper](https://arxiv.org/pdf/1909.10837.pdf)]
  [[code](https://github.com/zbs881314/Temporal-Coded-Deep-SNN)]
* Hao Wu, Yueyi Zhang, ... ***AAAI 2021***<br />
"Training Spiking Neural Networks with Accumulated Spiking Flow"
 [[paper](https://www.aaai.org/AAAI21Papers/AAAI-4138.WuHao.pdf)]
* Hanle Zheng, Yujie Wu, Lei Deng, Yifan Hu, Guoqi Li ***AAAI 2021***<br />
"Going Deeper With Directly-Trained Larger Spiking Neural Networks"
 [[paper](https://arxiv.org/abs/2011.05280)]
 * Wenrui Zhang, Peng Li ***NIPS 2020***<br />
"Temporal Spike Sequence Learning via Backpropagation for Deep Spiking Neural Networks"
 [[paper](https://arxiv.org/abs/2002.10085)]
 [[code](https://github.com/stonezwr/TSSL-BP)]
* Jinseok Kim, Kyungsu Kim, Jae-Joon Kim ***NIPS 2020***<br />
"Unifying Activation- and Timing-based Learning Rules for Spiking Neural Networks"
 [[paper](https://papers.nips.cc/paper/2020/file/e2e5096d574976e8f115a8f1e0ffb52b-Paper.pdf)]
 [[code](https://github.com/KyungsuKim42/ANTLR)]
 * Qianyi Li, Cengiz Pehlevan ***NIPS 2020***<br />
"Minimax Dynamics of Optimally Balanced Spiking Networks of Excitatory and Inhibitory Neurons"
 [[paper](https://arxiv.org/abs/2006.08115)]
* Haowen Fang, Amar Shrestha, Ziyi Zhao, Qinru Qiu ***IJCAI 2020***<br />
"Exploiting Neuron and Synapse Filter Dynamics in Spatial Temporal Learning of Deep Spiking Neural Network"
 [[paper](https://www.ijcai.org/Proceedings/2020/0388.pdf)]
 [[code](https://github.com/Snow-Crash/snn-iir)]
 * Xiang Cheng, Yunzhe Hao, Jiaming Xu, Bo Xu ***IJCAI 2020***<br />
**LISNN**: "Improving Spiking Neural Networks with Lateral Interactions for Robust Object Recognition"
 [[paper](https://www.ijcai.org/Proceedings/2020/0211.pdf)]
 * Johannes C. Thiele, Olivier Bichler, Antoine Dupret ***ICLR 2020***<br />
"SpikeGrad: An ANN-equivalent Computation Model for Implementing Backpropagation with Spikes".
 [[paper](https://arxiv.org/pdf/1906.00851.pdf)]
 * Jordan Guerguiev, Konrad P. Körding, Blake A. Richards ***ICLR 2020*** <br />
"Spike-based causal inference for weight alignment".
 [[paper](https://arxiv.org/pdf/1910.01689.pdf)]
 [[code](https://anonfile.com/51V8Ge66n3/Code_zip)]
  * Kian Hamedani, Lingjia Liu, Shiya Liu, Haibo He, Yang Yi ***AAAI 2020***<br />
"Deep Spiking Delayed Feedback Reservoirs and Its Application in Spectrum Sensing of MIMO-OFDM Dynamic Spectrum Sharing"
 [[paper](https://aaai.org/ojs/index.php/AAAI/article/view/5484)]
 * Wenrui Zhang, Peng Li ***NIPS 2019***<br />
"Spike-Train Level Backpropagation for Training Deep Recurrent Spiking Neural Networks".
 [[paper](http://papers.nips.cc/paper/8995-spike-train-level-backpropagation-for-training-deep-recurrent-spiking-neural-networks)]
* Yujie Wu, Lei Deng, Guoqi Li, Jun Zhu, Yuan Xie, Luping Shi ***AAAI 2019***<br />
"Direct Training for Spiking Neural Networks: Faster, Larger, Better".
 [[paper](https://aaai.org/ojs/index.php/AAAI/article/view/3929)]
 [[code]( https://github.com/yjwu17/BP-for-SpikingNN)]
 * Malu Zhang, Jibin Wu, Yansong Chua, Xiaoling Luo, Zihan Pan, Dan Liu, Haizhou Li ***AAAI 2019***<br />
**MPD-AL** "An Efficient Membrane Potential Driven Aggregate-Label Learning Algorithm for Spiking Neurons".
 [[paper](https://aaai.org/ojs/index.php/AAAI/article/view/3932)]
* Cengiz Pehlevan ***ICASSP 2019***<br />
"A Spiking Neural Network with Local Learning Rules Derived from Nonnegative Similarity Matching".
 [[paper](https://ieeexplore.ieee.org/document/8682290)]
* Megumi Ito, Malte J. Rasch, Masatoshi Ishii, Atsuya Okazaki, SangBum Kim, Junka Okazawa, Akiyo Nomura, Kohji Hosokawa, Wilfried Haensch***ICONIP 2019*** <br />
"Training Large-Scale Spiking Neural Networks on Multi-core Neuromorphic System Using Backpropagation".
 [[paper](https://link.springer.com/chapter/10.1007%2F978-3-030-36718-3_16)]
 * Johannes C. Thiele, Olivier Bichler, Antoine Dupret, Sergio Solinas, Giacomo Indiveri ***IJCNN 2019***<br />
"A Spiking Network for Inference of Relations Trained with Neuromorphic Backpropagation".
 [[paper](https://ieeexplore.ieee.org/document/8852360)]
  * Thomas Miconi, Jeff Clune, Kenneth O. Stanley ***ICML 2018***<br />
"Differentiable plasticity: training plastic neural networks with backpropagation".
 [[paper](https://arxiv.org/abs/1804.02464)]
  [[code]( https://github.com/uber-research/differentiable-plasticity)]
 * 	Dongsung Huh, Terrence J. Sejnowski ***NIPS 2018***<br />
"Gradient Descent for Spiking Neural Networks".
 [[paper](http://papers.nips.cc/paper/7417-gradient-descent-for-spiking-neural-networks)]
* Yingyezhe Jin, Wenrui Zhang, Peng Li ***NIPS 2018***<br />
"Hybrid Macro/Micro Level Backpropagation for Training Deep Spiking Neural Networks".
 [[paper](http://papers.nips.cc/paper/7932-hybrid-macromicro-level-backpropagation-for-training-deep-spiking-neural-networks)]
 
### SNN trained with Biological Plasticity Rules (STDP, Hebbian,etc)
* Chankyu Lee, Adarsh Kumar Kosta, Alex Zihao Zhu, Kenneth Chaney, Kostas Daniilidis, Kaushik Roy ***ECCV 2020***<br />
"Spike-FlowNet: Event-based Optical Flow Estimation with Energy-Efficient Hybrid Neural Networks"
[[paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/html/6736_ECCV_2020_paper.php)]
[[code](https://github.com/chan8972/Spike-FlowNet)]
* Lin Zhu, Siwei Dong, Jianing Li, Tiejun Huang, Yonghong Tian ***CVPR 2020***<br />
"Retina-Like Visual Image Reconstruction via Spiking Neural Model"
[[paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhu_Retina-Like_Visual_Image_Reconstruction_via_Spiking_Neural_Model_CVPR_2020_paper.html)]
* Qianhui Liu, Haibo Ruan, Dong Xing, Huajin Tang, Gang Pan ***AAAI 2020***<br />
"Effective AER Object Classification Using Segmented Probability-Maximization
Learning in Spiking Neural Networks" AAAI (2020 **Oral**).
 [[paper](https://aaai.org/Papers/AAAI/2020GB/AAAI-LiuQ.6583.pdf)]
* Zuozhu Liu, Thiparat Chotibut, Christopher Hillar, Shaowei Lin ***AAAI 2020*** <br />
"Biologically Plausible Sequence Learning with Spiking Neural Networks"
 [[paper](https://arxiv.org/abs/1911.10943)]
 [[code](https://github.com/owen94/MPNets)]
 * Shenglan Li, Qiang Yu ***AAAI 2020***<br />
"New Efficient Multi-Spike Learning for Fast Processing and Robust Learning"
 [[paper](https://aaai.org/ojs/index.php/AAAI/article/view/5896)]
 * Pengjie Gu, Rong Xiao, Gang Pan, Huajin Tang ***IJCAI 2019***<br />
**STCA:** "STCA: Spatio-Temporal Credit Assignment with Delayed Feedback in Deep Spiking Neural Networks".
 [[paper](https://www.ijcai.org/proceedings/2019/0189.pdf)]
 [[code](https://github.com/Squirtle-gpj/STCA-DSNN)]
 * Rong Xiao, Qiang Yu, Rui Yan, Huajin Tang ***IJCAI 2019***<br />
"Fast and Accurate Classification with a Multi-Spike Learning Algorithm for Spiking Neurons".
 [[paper](https://www.ijcai.org/Proceedings/2019/0200.pdf)]
  * Lakshay Sahni, Debasrita Chakraborty, Ashish Ghosh ***AAAI 2019***<br />
"Implementation of Boolean AND and OR Logic Gates with Biologically Reasonable Time Constants in Spiking Neural Networks".
 [[paper](https://aaai.org/ojs/index.php/AAAI/article/view/5147)]
  * Robert Luke, David McAlpine ***ICASSP 2019***<br />
"A Spiking Neural Network Approach to Auditory Source Lateralisation".
 [[paper](https://ieeexplore.ieee.org/document/8683767)]
  * Hui Yan, Xinle Liu, Hong Huo, Tao Fang ***ICONIP 2019***<br />
"Mechanisms of Reward-Modulated STDP and Winner-Take-All in Bayesian Spiking Decision-Making Circuit".
 [[paper](https://link.springer.com/chapter/10.1007%2F978-3-030-36718-3_14)]
  * Yanli Yao, Qiang Yu, Longbiao Wang, Jianwu Dang ***IJCNN 2019***<br />
"A Spiking Neural Network with Distributed Keypoint Encoding for Robust Sound Recognition".
 [[paper](https://ieeexplore.ieee.org/document/8852166)]
  * Pierre Falez, Pierre Tirilly, Ioan Marius Bilasco, Philippe Devienne, Pierre Boulet ***IJCNN 2019*** <br />
"Multi-layered Spiking Neural Network with Target Timestamp Threshold Adaptation and STDP".
 [[paper](https://ieeexplore.ieee.org/document/8852346)]
 * Jibin Wu, Yansong Chua, Malu Zhang, Qu Yang, Guoqi Li, Haizhou Li ***IJCNN 2019***<br />
"Deep Spiking Neural Network with Spike Count based Learning Rule".
 [[paper](https://ieeexplore.ieee.org/document/8852380)]
  * Maximilian P. R. Löhr, Daniel Schmid, Heiko Neumann ***IJCNN 2019***<br />
"Motion Integration and Disambiguation by Spiking V1-MT-MSTl Feedforward-Feedback Interaction".
 [[paper](https://ieeexplore.ieee.org/document/8852029)]
   * Esma Mansouri-Benssassi, Juan Ye ***IJCNN 2019***<br />
"Speech Emotion Recognition With Early Visual Cross-modal Enhancement Using Spiking Neural Networks".
 [[paper](https://ieeexplore.ieee.org/document/8852473)]
 * 	Mikhail Kiselev, Andrey Lavrentyev ***IJCNN 2019***<br />
"A Preprocessing Layer in Spiking Neural Networks - Structure, Parameters, Performance Criteria".
 [[paper](https://ieeexplore.ieee.org/document/8851848)]
 *Won-Mook Kang, Chul-Heung Kim, Soochang Lee, Sung Yun Woo, Jong-Ho Bae, Byung-Gook Park, Jong-Ho Lee ***IJCNN 2019***<br />
"A Spiking Neural Network with a Global Self-Controller for Unsupervised Learning Based on Spike-Timing-Dependent Plasticity Using Flash Memory Synaptic Devices".
 [[paper](https://ieeexplore.ieee.org/document/8851744)]
  * Lyes Khacef, Benoît Miramond, Diego Barrientos, Andres Upegui ***IJCNN 2019***<br />
"Self-organizing neurons: toward brain-inspired unsupervised learning".
 [[paper](https://ieeexplore.ieee.org/document/8852098)]
 * 	Peter O'Connor, Efstratios Gavves, Matthias Reisser, Max Welling ***ICLR 2018***<br />
"Temporally Efficient Deep Learning with Spikes".
 [[paper](https://openreview.net/forum?id=HkZy-bW0-)]
 * 	Aditya Gilra, Wulfram Gerstner ***ICML 2018***<br />
"Non-Linear Motor Control by Local Learning in Spiking Neural Networks".
 [[paper](http://proceedings.mlr.press/v80/gilra18a.html)]
 *Guillaume Bellec, Darjan Salaj, Anand Subramoney, Robert A. Legenstein, Wolfgang Maass ***NIPS 2018***<br />
"Long short-term memory and Learning-to-learn in networks of spiking neurons".
 [[paper](http://papers.nips.cc/paper/7359-long-short-term-memory-and-learning-to-learn-in-networks-of-spiking-neurons)]
 * Sumit Bam Shrestha, Garrick Orchard ***NIPS 2018***<br />
**SLAYER** "Spike Layer Error Reassignment in Time".
 [[paper](http://papers.nips.cc/paper/7415-slayer-spike-layer-error-reassignment-in-time)]
 [[code]( https://github.com/bamsumit/slayerPytorch)]
 * Yu Qi, Jiangrong Shen, Yueming Wang, Huajin Tang, Hang Yu, Zhaohui Wu, Gang Pan ***IJCAI 2018***<br />
Jointly Learning Network Connections and Link Weights in Spiking Neural Networks".
 [[paper](https://www.ijcai.org/Proceedings/2018/221)]
* Qi Xu, Yu Qi, Hang Yu, Jiangrong Shen, Huajin Tang, Gang Pan ***IJCAI 2018***<br />
**CSNN** "An Augmented Spiking based Framework with Perceptron-Inception".
 [[paper](https://www.ijcai.org/Proceedings/2018/228)]
 * Tielin Zhang, Yi Zeng, Dongcheng Zhao, Bo Xu ***IJCAI 2018***<br />
**VPSNN** Tielin Zhang, Yi Zeng, Dongcheng Zhao, Bo Xu.
 [[paper](https://www.ijcai.org/Proceedings/2018/229)]
  * Alireza Alemi, Christian K. Machens, Sophie Denève, Jean-Jacques E. Slotine ***AAAI 2018***<br />
"Learning Nonlinear Dynamics in Efficient, Balanced Spiking Networks Using Local Plasticity Rules".
 [[paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17438)]
 * Tielin Zhang, Yi Zeng, Dongcheng Zhao, Mengting Shi ***AAAI 2018***<br />
"A Plasticity-Centric Approach to Train the Non-Differential Spiking Neural Networks".
 [[paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16105)]
  * Alireza Bagheri, Osvaldo Simeone, Bipin Rajendran ***ICASSP 2018***<br />
"Training Probabilistic Spiking Neural Networks with First- To-Spike Decoding".
 [[paper](https://arxiv.org/pdf/1710.10704.pdf)]
   * Qiang Yu, Longbiao Wang, Jianwu Dang ***ICONIP 2018***<br />
"Efficient Multi-spike Learning with Tempotron-Like LTP and PSD-Like LTD".
 [[paper](https://doi.org/10.1007/978-3-030-04167-0_49)]
  * Jiaxing Liu, Guoping Zhao ***IJCNN 2018***<br />
"A bio-inspired SOSNN model for object recognition".
 [[paper](https://ieeexplore.ieee.org/document/8489076/)]
   * Amirhossein Tavanaei, Zachary Kirby, Anthony S. Maida ***IJCNN 2018***<br />
"Training Spiking ConvNets by STDP and Gradient Descent".
 [[paper](https://ieeexplore.ieee.org/document/8489104)]
   * Yu Miao, Huajin Tang, Gang Pan ***IJCNN 2018***<br />
"A Supervised Multi-Spike Learning Algorithm for Spiking Neural Networks".
 [[paper](https://ieeexplore.ieee.org/document/8489175)]
 * Timoleon Moraitis, Abu Sebastian, Evangelos Eleftheriou ***IJCNN 2018***<br />
"Spiking Neural Networks Enable Two-Dimensional Neurons and Unsupervised Multi-Timescale Learning".
 [[paper](https://ieeexplore.ieee.org/document/8489218)]
  * Sam Slade, Li Zhang ***IJCNN 2018***<br />
"Topological Evolution of Spiking Neural Networks".
 [[paper](https://ieeexplore.ieee.org/document/8489375)]
  *	Ruizhi Chen, Hong Ma, Peng Guo, Shaolin Xie, Pin Li, Donglin Wang ***IJCNN 2018***<br />
"Low Latency Spiking ConvNets with Restricted Output Training and False Spike Inhibition".
 [[paper](https://ieeexplore.ieee.org/document/8489400)]
 * Pierre Falez, Pierre Tirilly, Ioan Marius Bilasco, Philippe Devienne, Pierre Boulet ***IJCNN 2018***<br />
"Mastering the Output Frequency in Spiking Neural Networks".
 [[paper](https://ieeexplore.ieee.org/document/8489410)]
  *	Daqi Liu, Shigang Yue ***IJCNN 2018***<br />
"Video-Based Disguise Face Recognition Based on Deep Spiking Neural Network".
 [[paper](https://ieeexplore.ieee.org/document/8489476)]
  * Johannes C. Thiele, Olivier Bichler, Antoine Dupret ***IJCNN 2018***<br />
"A Timescale Invariant STDP-Based Spiking Deep Network for Unsupervised Online Feature Extraction from Event-Based Sensor Data".
 [[paper](https://ieeexplore.ieee.org/document/8489666)]
  * Hananel Hazan, Daniel J. Saunders, Darpan T. Sanghavi, Hava T. Siegelmann, Robert Kozma ***IJCNN 2018***<br />
"Unsupervised Learning with Self-Organizing Spiking Neural Networks".
 [[paper](https://ieeexplore.ieee.org/document/8489673)]
  * Daniel J. Saunders, Hava T. Siegelmann, Robert Kozma, Miklós Ruszinkó ***IJCNN 2018***<br />
"STDP Learning of Image Patches with Convolutional Spiking Neural Networks".
 [[paper](https://ieeexplore.ieee.org/document/8489684)]
   *	Jibin Wu, Yansong Chua, Haizhou Li ***IJCNN 2018***<br />
"A Biologically Plausible Speech Recognition Framework Based on Spiking Neural Networks".
 [[paper](https://ieeexplore.ieee.org/document/8489535)]
 * Antonio Jimeno-Yepes, Jianbin Tang, Benjamin Scott Mashford ***IJCAI 2017***<br />
"Improving Classification Accuracy of Feedforward Neural Networks for Spiking Neuromorphic Chips".
 [[paper](https://www.ijcai.org/Proceedings/2017/274)]
  * Zhanhao Hu, Tao Wang, Xiaolin Hu ***ICONIP 2017***<br />
"An STDP-Based Supervised Learning Algorithm for Spiking Neural Network".
 [[paper](https://link.springer.com/chapter/10.1007%2F978-3-319-70096-0_10)]
   * Lin Zuo, Shan Chen, Hong Qu, Malu Zhang ***ICONIP 2017***<br />
"A Fast Precise-Spike and Weight-Comparison Based Learning Approach for Evolving Spiking Neural Networks".
 [[paper](https://link.springer.com/chapter/10.1007%2F978-3-319-70090-8_81)]
   * Amirhossein Tavanaei, Anthony S. Maida ***IJCNN 2017***<br />
"Multi-layer unsupervised learning in a spiking convolutional neural network".
 [[paper](https://ieeexplore.ieee.org/document/7966099)]
   * Takashi Matsubara ***IJCNN 2017***<br />
"Spike timing-dependent conduction delay learning model classifying spatio-temporal spike patterns".
 [[paper](https://ieeexplore.ieee.org/document/7966073)]
   * Laxmi R. Iyer, Arindam Basu ***IJCNN 2017***<br />
"Unsupervised learning of event-based image recordings using spike-timing-dependent plasticity".
 [[paper](https://ieeexplore.ieee.org/document/7966074)]
 * 	Gopalakrishnan Srinivasan, Sourjya Roy, Vijay Raghunathan, Kaushik Roy ***IJCNN 2017***<br />
"Spike timing dependent plasticity based enhanced self-learning for efficient pattern recognition in spiking neural networks".
 [[paper](https://ieeexplore.ieee.org/document/7966075)]
 * Amar Shrestha, Khadeer Ahmed, Yanzhi Wang, Qinru Qiu ***IJCNN 2017***<br />
"Stable spike-timing dependent plasticity rule for multilayer unsupervised and supervised learning".
 [[paper](https://ieeexplore.ieee.org/document/7966096)]
  * Timoleon Moraitis, Abu Sebastian, Irem Boybat, Manuel Le Gallo, Tomas Tuma, Evangelos Eleftheriou ***IJCNN 2017***<br />
"Fatiguing STDP: Learning from spike-timing codes in the presence of rate codes".
 [[paper](https://ieeexplore.ieee.org/document/7966072)]
   * Yingyezhe Jin, Peng Li ***IJCNN 2017***<br />
"Calcium-modulated supervised spike-timing-dependent plasticity for readout training and sparsification of the liquid state machine".
 [[paper](https://ieeexplore.ieee.org/document/7966097)]
 * 	Amirali Amirsoleimani, Majid Ahmadi, Arash Ahmadi ***IJCNN 2017***<br />
"STDP-based unsupervised learning of memristive spiking neural network by Morris-Lecar model".
 [[paper](https://ieeexplore.ieee.org/document/7966284)]