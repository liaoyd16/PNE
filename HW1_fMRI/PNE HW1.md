### PNE HW1

---

计62 廖远达 2016011245

####问题

1. 人脑结构左右不对称性分析

- 1.1 问题

  a) 计算各脑区皮层面积占比
  b) 统计受试左右大脑半球中，transverse-temporal，pars-opercularis两个脑区的灰质体积比，并计算不对称指数
  c) 将左右海马成对显示在三维空间中

- 1.2方法

  首先分析本问题的数据，然后分析如何计算并呈现为有意义的大脑属性。

  1.2.1 实验数据描述

  本问题的数据是通过MRI采集的人脑皮层类型分布，这个分布的参数有x,y,z。因此，在.mat文件中使用了ndims=3的矩阵表示的这一分布，每一个高维“像素”的体积为$1mm^3​$，数值为皮层标签。

  1.2.2 a) 问题属性的计算

  为了计算各脑区的皮层面积，需要知道这些标签哪些属于灰质（即皮层），属于四个脑区的哪一个。为此，在FreeSurfer提供的标签文档(FsTutorial_AnatomicalROI_FreeSurferColorLUT.txt)中，查询所有含ctx(cortext)的标签(label)，并查询这些标签对应的脑区(lobe)。

  随后，遍历整个空间的voxel，统计有多少voxel属于感兴趣的脑区所含的标签，即可统计出该被试这个脑区皮层的表面积。再计算该脑区表面积在所有表面积中的占比即可。

  （注，此处未统计边缘脑区(limbic lobe)的表面积）

  |         | left-hemisphere | right-hemisphere |
  | ------- | --------------- | ---------------- |
  | indices | 1000~1035       | 2000~2035        |

  1.2.3 b) 问题的计算

  为计算某一被试的，某一脑区的灰质体积，使用标签定位这些灰质所在的voxel，并统计所有这些voxel的总体积即可。

  根据感兴趣区域的属性：灰质、transverse-temporal/pars-opercularis、hemisphere，找到标签如下：

  |      |left-hemisphere|right-hemisphere|
  | ---- | ---- | ---- |
  |transverse-temporal|ctx-lh-transversetemporal(1034)| ctx-rh-transversetemporal(2034) |
  |pars-opercularis| ctx-lh-parsopercularis(1019) | ctx-rh-parsopercularis(2019) |

  不对称性指数的计算为(L-R)/[(L+R)/2]，计算可得某一被试、某一脑区的不对称性指数。

  1.2.4 c) 问题的计算

  为了可视化某个大脑的海马结构，并成对显示，选用了matplotlib中的voxels()函数进行3D交互式绘图。海马体的标签查询如下：

  |       | left-hemisphere | right-hemisphere |
  | ----- | --------------- | ---------------- |
  | label | 17              | 53               |

- 结果

  a)

  - brain1:

    frontal=34.17%
    temporal=23.80%
    parietal=26.91%
    occipital=10.15%

  - brain2:

    frontal=35.55%
    temporal=23.51%
    parietal=27.70%
    occipital=8.57%

  b) 

  |        | transverse-temporal | pars-opercularis |
  | ------ | ------------------- | ---------------- |
  | brain1 | -0.235              | -0.898           |
  | brain2 | -0.344              | -0.849           |

  c)

![Figure_1](/Users/liaoyuanda/Desktop/PNE/hw1/submit/Figure_1.png)

 	被试1
 	
  ![Figure_2](/Users/liaoyuanda/Desktop/PNE/hw1/submit/Figure_2.png)

  	被试2

- 讨论

  上述属性中，我们基于参考文献[1]对脑区左右不对称的可能原因进行讨论。

  基于本问题处理的数据，我们发现两位被试的不对称性指数均为负；两位被试的不对称性指数相差分别为37.6%, 5.6%。

  再分析两个脑区的功能：transverse-temporal的功能与听觉关系密切，而pars-operculars与语言处理关系密切。

  根据参考文献[1]的发现，人脑的不对称性和性别、年龄、ICV、惯用手和遗传等有关([1]E5161)。考虑到大多数人是右手惯用手，惯用手不对称性和两个脑区的不对称性趋势相同，有可能是不对称性的原因之一；对于两个人的不对称性的差异，尤其是在transverse-temporal脑区的不对称性差异，有可能和某些轻度的病理和失常有关。

2. fMRI定位视觉运动处理脑区

- 2.1 问题

   a) 可视化fMRI数据
   b) 分析对视觉运动次级有显著激活的像素(voxel)，并在脑片上绘图

- 2.2 方法

   2.2.1 数据格式描述

   本问题的数据是通过fMRI采集的人脑血氧活动分布，这个分布的参数有x,y,z,t。因此，在bold_y.nii文件中使用了ndims=4的矩阵表示的这一分布，每一个高维“像素”的体积为$1(mm^3\times s)$，数值为浮点数值。

   2.2.2 a) 问题的计算与可视化

   使用matlab工具[6]进行可视化，见代码；

   2.2.3 b) 问题的计算与可视化

   为了计算对视觉运动刺激有显著激活的voxels，首先考察刺激的表示和voxel活动的表示。

   设voxel活动为$v = v(t)$，刺激为s = s(t)，实际知道的刺激序列为$s_k = s(kt)​$。

   我们设$v(t) = \beta_1 s(t) + \beta_2 + \epsilon_t$，近似地我们设$v_k=s_k\beta_1 + \beta_2 + \epsilon_k$，对每一个k，用矩阵形式重写为$v = X[\beta_1, \beta_2]^T + \epsilon= [s, ones][\beta_1, \beta_2]^T+\epsilon$，则使得$\beta_1$最大的$v_k = v(kt)$序列为与刺激最为显著相关的voxel。

   为求得$\beta_1$，我们计算$[\beta_1, \beta_2]^T$，通过伪逆公式$[\beta_1, \beta_2]^T = (X^TX)^{-1}X^T v$，计算得到$[\beta_1, \beta_2]^T$。

   一种改进的方法考虑了Bold函数的影响：bold函数将特定的voxel抽象成单位冲激响应响应为Bold函数的元件，我们将stimuli的信号卷积Bold窗函数，按照上述方法进行计算即可。由于采样的离散化，我们根据卷积的性质，知道用离散采样的Bold函数卷积stimuli即可得到连续函数卷积的采样。

   为可视化上述坐标点，希望在MNI坐标下标注出这些坐标点，并使用matplotlib工具[3]对切面图像进行可视化。

- 2.3 结果

   a)

   ![2_1](/Users/liaoyuanda/Desktop/PNE/hw1/submit/2_1.jpeg)

   b)

   由于MNI-XYZ坐标转换出现问题，还是使用XYZ坐标下的矩阵进行可视化（从左至右依次为侧向、正向、垂直切面）：

   ![brain1_2_b](/Users/liaoyuanda/Desktop/PNE/hw1/submit/brain1_2_b.jpeg)

   ​	GLM

   ![brain2_2_b](/Users/liaoyuanda/Desktop/PNE/hw1/submit/brain2_2_b.jpeg)

   ​	GLM with HRF

   （上述为部分切片图像，其他图像运行'python funcs.py'即可运行）

- 2.4 讨论

    改进方法和原方法得到的活跃区域不同。大体的趋势是，HRF卷积方法得到的显著voxel与原GLM方法得到的区域相比处于更低级的通路上。这可能是因为较为低端的回路受到血氧动力学造成的非线性加窗影响更大。

#### 代码运行方法

- 建立新的文件夹temp，在其中放入twobrains.mat, bold_y.nii, t1_y.nii
- 在temp中放入压缩包，解压
- 更改\_\_init\_\_.py中的ROOT_DIR为temp的绝对路径
- 在压缩包中运行命令行指令 'python funcs.py'

#### 参考文献

[1] Kong et al.Mapping cortical brain asymmetry in 17,141 healthy
individuals worldwide via the ENIGMA Consortium.PNAS;115(22):E5154-
E5163.2018

[2] Born RT, Bradley DC. Structure and function of visual area MT. Annu
Rev Neurosci. 2005;28:157-89.

[3] Hunter, J. D., Matplotlib: A 2D graphics environment. Computing In Science \& Engineering. 2007

[4] MATLAB and Statistics Toolbox Release 2016b, The MathWorks, Inc., Natick, Massachusetts, United States.

[5] NeuroImaging Tools and Resources Collaboratory, mni2orfromxyz package, https://www.nitrc.org/frs/?group_id=477&release_id=1762#

[6] MATLAB neural image visualising tools, https://cn.mathworks.com/matlabcentral/fileexchange/8797