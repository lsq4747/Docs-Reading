# AutoTVM

[toc]

AutoTVM与AutoScheduler都是**基于搜索**的自动优化，主要包括：

* cost model&feature extraction

* cost model的构建，一种储存程序**benchmark**结果的记录格式

  > **benchmark log format**
  >
  > ```json
  > {
  >   "workload":"arcface_resnet100",
  >   "engine":"tvm",
  >   "hardware":"gcp-c2-standard-16",
  >   "runtime_ms_mean":109.43004820081924,
  >   "runtime_ms_std":0.09078385126800587,
  >   "timestamp":"20191123003411",
  >   "schema_version":"0.1",
  >   "metadata":{
  >     "docker_tag":"tvmai/ci-gpu:v0.53"
  >   },
  >   "workload_args":{
  >     "input_shape_dict":{
  >       "data":[
  >         1,
  >         3,
  >         112,
  >         112
  >       ]
  >     },
  >     "input_type_dict":{
  >       "data":"float32"
  >     },
  >     "input_value_dict":{}
  >   },
  >   "workload_metadata":{
  >     "class":"vision",
  >     "doc_url":"https://github.com/onnx/models/blob/master/vision/body_analysis/arcface/README.md",
  >     "md5":"66074b860f905295aab5a842be57f37d",
  >     "opset":8,
  >     "type":"body_analysis",
  >     "url":"https://s3.amazonaws.com/onnx-model-zoo/arcface/resnet100/resnet100.tar.gz"
  >   },
  >   "engine_version":"1.0.0",
  >   "engine_config":{},
  >   "compilation_config":{
  >     "relay_opt_level": 3
  >   },
  >   "software_config":{
  >     "os":"ubuntu:18.04",
  >     "pip":{
  >       "docker":"4.1.0",
  >       "gitpython":"3.0.4",
  >       "numpy":"1.17.4",
  >       "onnx":"1.6.0"
  >     }
  >   },
  >   "runtime_config":{},
  >   "hardware_config":{
  >     "cloud_machine_type":"c2-standard-16",
  >     "cloud_provider":"GCP",
  >     "cpu_count":16,
  >     "cpu_platform":"Intel Cascade Lake",
  >     "memory_GB":64
  >   },
  >   "execution_config":{},
  >   "metrics":{}
  > }
  > ```
  >
  > 

* 搜索策略

## Writing tunable template and Using auto-tuner

* 加载依赖
* 定义搜索空间
* 搜索搜索空间 

### 1: Define the search space

以blocked matrix multiplication为例：

```python
# Matmul V0: Constant tiling factor
def matmul_v0(N, L, M, dtype):
    A = te.placeholder((N, L), name='A', dtype=dtype)
    B = te.placeholder((L, M), name='B', dtype=dtype)

    k = te.reduce_axis((0, L), name='k')
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = te.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    yo, yi = s[C].split(y, 8)
    xo, xi = s[C].split(x, 8)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]
```

其中的tiling factor为8，但是这个值是取决于硬件环境和输入的，我们可以根据这些来选取tiling factor的最佳值。

在autoTVM里，我们定义一个可以调整的参数，或者叫做<kbd>knob</kbd>。

```python
# Matmul V1: List candidate values
@autotvm.template("tutorial/matmul_v1")  # 1. use a decorator
def matmul_v1(N, L, M, dtype):
    A = te.placeholder((N, L), name='A', dtype=dtype)
    B = te.placeholder((L, M), name='B', dtype=dtype)

    k = te.reduce_axis((0, L), name='k')
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = te.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    # 2. get the config object
    cfg = autotvm.get_config()

    # 3. define search space
    cfg.define_knob("tile_y", [1, 2, 4, 8, 16])
    cfg.define_knob("tile_x", [1, 2, 4, 8, 16])

    # 4. schedule according to config
    yo, yi = s[C].split(y, cfg['tile_y'].val)
    xo, xi = s[C].split(x, cfg['tile_x'].val)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]
```

这里搜索空间的大小为5x5

除了把knob可能的值都列出来之外，可以使用<kbd>ConfigSpace.define_split</kbd>来定义一个 split knob，它将列举所有可能的构造空间的方式。除此之外，还有<kbd>ConfigSpace.define_reorder</kbd>、<kbd>ConfigSpace.define_annotate</kbd>等。

```python
@autotvm.template("tutorial/matmul")
def matmul(N, L, M, dtype):
    A = te.placeholder((N, L), name='A', dtype=dtype)
    B = te.placeholder((L, M), name='B', dtype=dtype)

    k = te.reduce_axis((0, L), name='k')
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = te.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    ##### define space begin #####
    cfg = autotvm.get_config()
    cfg.define_split("tile_y", y, num_outputs=2)
    cfg.define_split("tile_x", x, num_outputs=2)
    ##### define space end #####

    # schedule according to config
    yo, yi = cfg["tile_y"].apply(s, C, y)
    xo, xi = cfg["tile_x"].apply(s, C, x)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]
```

### 2: Search through the space

在上一步中，我们建立了搜索空间。autoTVM中提到了四种不同策略的tuner：

* RandomTuner 随机顺序枚举
* GridSearchTuner 网格搜索（<1000)
* GATuner 遗传算法搜索
* ***XGBTuner*** 训练一个XGBoost模型来预测，并根据结果选下一批(10^9，CUDA GPU上一个conv2d算子的空间大小)

```python
# logging config (for printing tuning log to the screen)
logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

# There are two steps for measuring a config: build and run.
# By default, we use all CPU cores to compile program. Then measure them sequentially.
# We measure 5 times and take average to reduce variance.
measure_option = autotvm.measure_option(
    builder='local',
    runner=autotvm.LocalRunner(number=5))

# Begin tuning with RandomTuner, log records to file `matmul.log`
# You can use alternatives like XGBTuner.
tuner = autotvm.tuner.RandomTuner(task)
tuner.tune(n_trial=10,
           measure_option=measure_option,
           callbacks=[autotvm.callback.log_to_file('matmul.log')])
```

## Blocked Matrix Multiplication

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 23:06:04 2020

@author: lsq
"""


import logging
import sys

import numpy as np
import tvm

#导入autotvm模块
from tvm import autotvm
from tvm import te

N, L, M = 512, 512, 512
dtype = "float32"

def matmul_v0(N, L, M, dtype):
    A = te.placeholder((N, L), name='A', dtype=dtype)
    B = te.placeholder((L, M), name='B', dtype=dtype)
    
    k = te.reduce_axis((0,L), name='k')
    C = te.compute((N,M), lambda i,j: te.sum(A[i,k]*B[k,j],axis=k), name='C')
    s = te.create_schedule(C.op)
    
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]
    
    yo, yi = s[C].split(y, 8)
    xo, xi = s[C].split(x, 8)
    
    s[C].reorder(yo, xo, k, yi, xi)
    return s, [A, B, C]

@autotvm.template("matmul_v1")
def matmul_v1(N, L, M, dtype):
    A = te.placeholder((N, L), name='A', dtype=dtype)
    B = te.placeholder((L, M), name='B', dtype=dtype)

    k = te.reduce_axis((0, L), name='k')
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = te.create_schedule(C.op)
    
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]
    
    #获取autotvm配置对象
    cfg = autotvm.get_config()
    
    #定义搜索空间knob
    cfg.define_knob("tile_y", [1,2,4,8,16])
    cfg.define_knob("tile_x", [1,2,4,8,16])
    
    #4.根据配置来调度
    yo, yi = s[C].split(y, cfg['tile_y'].val)
    xo, xi = s[C].split(x, cfg['tile_x'].val)
    
    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]


@autotvm.template("matmul")
def matmul(N, L, M, dtype):
    A = te.placeholder((N, L), name='A', dtype=dtype)
    B = te.placeholder((L, M), name='B', dtype=dtype)

    k = te.reduce_axis((0, L), name='k')
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = te.create_schedule(C.op)
    
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]
    
    #获取autotvm配置对象
    cfg = autotvm.get_config()
    
    #定义搜索空间knob
    cfg.define_split("tile_y", y, num_outputs=2)
    cfg.define_split("tile_x", x, num_outputs=2)
    
    #4.根据配置来调度
    yo, yi = cfg["tile_y"].apply(s, C, y)
    xo, xi = cfg["tile_x"].apply(s, C, x)
    
    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]

task = autotvm.task.create("matmul_v1", args=(N, L, M, dtype), target='llvm')
print(task.config_space)

logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

#测量配置有两个步骤：构建和运行。
#默认情况下，我们使用所有CPU核来编译程序。然后按顺序测量它们。
#测量5次并取平均值来减少误差。
measure_option = autotvm.measure_option(builder='local', runner=autotvm.LocalRunner(number=5))

#开始调优，将日志记录到文件`matmul.log`
tuner = autotvm.tuner.GridSearchTuner(task)
tuner.tune(n_trial=100,
           measure_option=measure_option,
           callbacks=[autotvm.callback.log_to_file('matmul_v1.log')])

dispatch_context = autotvm.apply_history_best('matmul_v1.log')
best_config = dispatch_context.query(task.target, task.workload)
print("\nBest config:")
print(best_config)

```

**matmul_v1**的运行结果

```
ConfigSpace (len=25, space_map=
   0 tile_y: OtherOption([1, 2, 4, 8, 16]) len=5
   1 tile_x: OtherOption([1, 2, 4, 8, 16]) len=5
)
Get devices for measurement successfully!
No: 1   GFLOPS: 0.50/0.50       result: MeasureResult(costs=(0.5356015732,), error_no=0, all_cost=8.86544418334961, timestamp=1598974852.889051)        [('tile_y', 1), ('tile_x', 1)],None,0
No: 2   GFLOPS: 0.72/0.72       result: MeasureResult(costs=(0.3728112216,), error_no=0, all_cost=6.271931409835815, timestamp=1598974859.155425)       [('tile_y', 2), ('tile_x', 1)],None,1
No: 3   GFLOPS: 0.93/0.93       result: MeasureResult(costs=(0.2879593816,), error_no=0, all_cost=4.897149324417114, timestamp=1598974864.0440013)      [('tile_y', 4), ('tile_x', 1)],None,2
No: 4   GFLOPS: 0.91/0.93       result: MeasureResult(costs=(0.29556170119999997,), error_no=0, all_cost=5.003795385360718, timestamp=1598974869.0451303)       [('tile_y', 8), ('tile_x', 1)],None,3
No: 5   GFLOPS: 0.65/0.93       result: MeasureResult(costs=(0.4110976536,), error_no=0, all_cost=6.8430399894714355, timestamp=1598974875.8721173)     [('tile_y', 16), ('tile_x', 1)],None,4
No: 6   GFLOPS: 0.72/0.93       result: MeasureResult(costs=(0.3753784752,), error_no=0, all_cost=6.304967164993286, timestamp=1598974882.1750755)      [('tile_y', 1), ('tile_x', 2)],None,5
No: 7   GFLOPS: 0.93/0.93       result: MeasureResult(costs=(0.2888192758,), error_no=0, all_cost=4.903738498687744, timestamp=1598974887.0929942)      [('tile_y', 2), ('tile_x', 2)],None,6
No: 8   GFLOPS: 1.11/1.11       result: MeasureResult(costs=(0.241006717,), error_no=0, all_cost=4.186624050140381, timestamp=1598974891.2639177)       [('tile_y', 4), ('tile_x', 2)],None,7
No: 9   GFLOPS: 1.15/1.15       result: MeasureResult(costs=(0.2340248862,), error_no=0, all_cost=4.1293041706085205, timestamp=1598974895.5744483)     [('tile_y', 8), ('tile_x', 2)],None,8
No: 10  GFLOPS: 1.11/1.15       result: MeasureResult(costs=(0.24117774500000003,), error_no=0, all_cost=4.228281497955322, timestamp=1598974899.7340603)       [('tile_y', 16), ('tile_x', 2)],None,9
No: 11  GFLOPS: 0.88/1.15       result: MeasureResult(costs=(0.3066449548,), error_no=0, all_cost=5.207779884338379, timestamp=1598974904.9307563)      [('tile_y', 1), ('tile_x', 4)],None,10
No: 12  GFLOPS: 1.10/1.15       result: MeasureResult(costs=(0.2432617286,), error_no=0, all_cost=4.23800802230835, timestamp=1598974909.1272433)       [('tile_y', 2), ('tile_x', 4)],None,11
No: 13  GFLOPS: 1.36/1.36       result: MeasureResult(costs=(0.1980989446,), error_no=0, all_cost=3.514873504638672, timestamp=1598974912.5978806)      [('tile_y', 4), ('tile_x', 4)],None,12
No: 14  GFLOPS: 1.42/1.42       result: MeasureResult(costs=(0.1891090528,), error_no=0, all_cost=3.38094162940979, timestamp=1598974915.9479918)       [('tile_y', 8), ('tile_x', 4)],None,13
No: 15  GFLOPS: 1.66/1.66       result: MeasureResult(costs=(0.1614592712,), error_no=0, all_cost=2.901059627532959, timestamp=1598974918.8308468)      [('tile_y', 16), ('tile_x', 4)],None,14
No: 16  GFLOPS: 1.05/1.66       result: MeasureResult(costs=(0.2549909032,), error_no=0, all_cost=4.402253150939941, timestamp=1598974923.2140615)      [('tile_y', 1), ('tile_x', 8)],None,15
No: 17  GFLOPS: 1.39/1.66       result: MeasureResult(costs=(0.1927918744,), error_no=0, all_cost=3.4512789249420166, timestamp=1598974926.857934)      [('tile_y', 2), ('tile_x', 8)],None,16
No: 18  GFLOPS: 1.42/1.66       result: MeasureResult(costs=(0.18893806959999998,), error_no=0, all_cost=3.3501639366149902, timestamp=1598974930.1765952)      [('tile_y', 4), ('tile_x', 8)],None,17
No: 19  GFLOPS: 1.89/1.89       result: MeasureResult(costs=(0.1422665278,), error_no=0, all_cost=2.6079344749450684, timestamp=1598974932.7515259)     [('tile_y', 8), ('tile_x', 8)],None,18
No: 20  GFLOPS: 2.14/2.14       result: MeasureResult(costs=(0.125315276,), error_no=0, all_cost=2.3372669219970703, timestamp=1598974935.0511827)      [('tile_y', 16), ('tile_x', 8)],None,19
No: 21  GFLOPS: 1.26/2.14       result: MeasureResult(costs=(0.2125106342,), error_no=0, all_cost=3.7211294174194336, timestamp=1598974938.7555583)     [('tile_y', 1), ('tile_x', 16)],None,20
No: 22  GFLOPS: 1.39/2.14       result: MeasureResult(costs=(0.1929266698,), error_no=0, all_cost=3.439847230911255, timestamp=1598974942.138658)       [('tile_y', 2), ('tile_x', 16)],None,21
No: 23  GFLOPS: 1.96/2.14       result: MeasureResult(costs=(0.1366422222,), error_no=0, all_cost=2.535428285598755, timestamp=1598974944.6226578)      [('tile_y', 4), ('tile_x', 16)],None,22
No: 24  GFLOPS: 2.33/2.33       result: MeasureResult(costs=(0.11533073840000001,), error_no=0, all_cost=2.1850152015686035, timestamp=1598974946.7703123)      [('tile_y', 8), ('tile_x', 16)],None,23
No: 25  GFLOPS: 2.37/2.37       result: MeasureResult(costs=(0.1130593862,), error_no=0, all_cost=2.0939857959747314, timestamp=1598974949.021378)      [('tile_y', 16), ('tile_x', 16)],None,24
Finish loading 45 records

Best config:
[('tile_y', 16), ('tile_x', 16)],None,24
```

改为用XGBTuner之后：

```
ConfigSpace (len=25, space_map=
   0 tile_y: OtherOption([1, 2, 4, 8, 16]) len=5
   1 tile_x: OtherOption([1, 2, 4, 8, 16]) len=5
)
Get devices for measurement successfully!
No: 1   GFLOPS: 2.33/2.33       result: MeasureResult(costs=(0.11507949740000001,), error_no=0, all_cost=2.2350687980651855, timestamp=1599016071.885539)       [('tile_y', 8), ('tile_x', 16)],None,23
No: 2   GFLOPS: 1.87/2.33       result: MeasureResult(costs=(0.1432292858,), error_no=0, all_cost=2.6357853412628174, timestamp=1599016074.479024)      [('tile_y', 8), ('tile_x', 8)],None,18
No: 3   GFLOPS: 1.25/2.33       result: MeasureResult(costs=(0.2155873162,), error_no=0, all_cost=3.7730910778045654, timestamp=1599016078.2151458)     [('tile_y', 1), ('tile_x', 16)],None,20
No: 4   GFLOPS: 1.43/2.33       result: MeasureResult(costs=(0.1882073092,), error_no=0, all_cost=3.3661375045776367, timestamp=1599016081.5415118)     [('tile_y', 4), ('tile_x', 8)],None,17
No: 5   GFLOPS: 0.67/2.33       result: MeasureResult(costs=(0.40114934719999995,), error_no=0, all_cost=6.76871132850647, timestamp=1599016088.2806637)        [('tile_y', 16), ('tile_x', 1)],None,4
No: 6   GFLOPS: 2.38/2.38       result: MeasureResult(costs=(0.11283017420000001,), error_no=0, all_cost=2.143136978149414, timestamp=1599016090.3765252)       [('tile_y', 16), ('tile_x', 16)],None,24
No: 7   GFLOPS: 1.97/2.38       result: MeasureResult(costs=(0.1365762726,), error_no=0, all_cost=2.5163557529449463, timestamp=1599016092.8552015)     [('tile_y', 4), ('tile_x', 16)],None,22
No: 8   GFLOPS: 0.90/2.38       result: MeasureResult(costs=(0.2972222028,), error_no=0, all_cost=5.072627544403076, timestamp=1599016097.903814)       [('tile_y', 1), ('tile_x', 4)],None,10
No: 9   GFLOPS: 1.41/2.38       result: MeasureResult(costs=(0.189903257,), error_no=0, all_cost=3.4131906032562256, timestamp=1599016101.5307906)      [('tile_y', 2), ('tile_x', 16)],None,21
No: 10  GFLOPS: 1.08/2.38       result: MeasureResult(costs=(0.24960683339999998,), error_no=0, all_cost=4.293562650680542, timestamp=1599016105.8129041)       [('tile_y', 2), ('tile_x', 4)],None,11
No: 11  GFLOPS: 0.93/2.38       result: MeasureResult(costs=(0.2874579496,), error_no=0, all_cost=4.887027025222778, timestamp=1599016110.6963203)      [('tile_y', 4), ('tile_x', 1)],None,2
No: 12  GFLOPS: 0.91/2.38       result: MeasureResult(costs=(0.2946190606,), error_no=0, all_cost=5.05925440788269, timestamp=1599016115.7376785)       [('tile_y', 2), ('tile_x', 2)],None,6
No: 13  GFLOPS: 1.40/2.38       result: MeasureResult(costs=(0.1920101482,), error_no=0, all_cost=3.4347591400146484, timestamp=1599016119.15792)       [('tile_y', 2), ('tile_x', 8)],None,16
No: 14  GFLOPS: 1.67/2.38       result: MeasureResult(costs=(0.1609213096,), error_no=0, all_cost=2.8957507610321045, timestamp=1599016122.044523)      [('tile_y', 16), ('tile_x', 4)],None,14
No: 15  GFLOPS: 0.67/2.38       result: MeasureResult(costs=(0.398075024,), error_no=0, all_cost=6.664667367935181, timestamp=1599016128.7063446)       [('tile_y', 1), ('tile_x', 2)],None,5
No: 16  GFLOPS: 0.61/2.38       result: MeasureResult(costs=(0.4425064658,), error_no=0, all_cost=7.3547422885894775, timestamp=1599016136.0870934)     [('tile_y', 1), ('tile_x', 1)],None,0
No: 17  GFLOPS: 0.72/2.38       result: MeasureResult(costs=(0.3732299752,), error_no=0, all_cost=6.227230072021484, timestamp=1599016142.530423)       [('tile_y', 2), ('tile_x', 1)],None,1
No: 18  GFLOPS: 1.07/2.38       result: MeasureResult(costs=(0.24988786319999998,), error_no=0, all_cost=4.287317752838135, timestamp=1599016146.8250325)       [('tile_y', 1), ('tile_x', 8)],None,15
No: 19  GFLOPS: 2.06/2.38       result: MeasureResult(costs=(0.1301685914,), error_no=0, all_cost=2.364795684814453, timestamp=1599016149.1972826)      [('tile_y', 16), ('tile_x', 8)],None,19
No: 20  GFLOPS: 1.36/2.38       result: MeasureResult(costs=(0.19746207600000001,), error_no=0, all_cost=3.466331958770752, timestamp=1599016152.6642995)       [('tile_y', 4), ('tile_x', 4)],None,12
Finish loading 65 records

Best config:
[('tile_y', 16), ('tile_x', 16)],None,24
```



##  Depthwise Convolution

https://tvm.apache.org/2017/08/22/Optimize-Deep-Learning-GPU-Operators-with-TVM-A-Depthwise-Convolution-Example

```python
# padding stage
PaddedInput = tvm.compute(
    (batch, in_channel, height_after_pad, width_after_pad),
    lambda b, c, i, j: tvm.select(
        tvm.all(i >= pad_top, i - pad_top < in_height, j >= pad_left, j - pad_left < in_width),
        Input[b, c, i - pad_top, j - pad_left], tvm.const(0.0)),
    name="PaddedInput")
# depthconv stage
di = tvm.reduce_axis((0, filter_height), name='di')
dj = tvm.reduce_axis((0, filter_width), name='dj')
Output = tvm.compute(
    (batch, out_channel, out_height, out_width),
    lambda b, c, i, j: tvm.sum(
        PaddedInput[b, c/channel_multiplier, i*stride_h + di, j*stride_w + dj] * Filter[c/channel_multiplier, c%channel_multiplier, di, dj],
        axis=[di, dj]),
    name='DepthwiseConv2d')

```



优化CUDA代码有三部分

* Data Reuse 

  因为从内存里取一次数据的成本高，故希望将输入数据加载到寄存器或者cache里重用。

  分为filter reuse和input reuse，input reuse通过tilling平铺实现

  <img  src= "https://tvm.apache.org/images/depthconv_tutorial/tiling.png">

* Shared Memory

  可以看作是GPU中的cache，是按块分配的。

  ![image](https://tvm.apache.org/images/depthconv_tutorial/GPU_memory_hierarchy.png)

* Bank Conflicts

  shared memory会面临bank conflicts，如果多个线程访问同一个内存bank，会造成bank conflicts。

  shared memory会将连续的地址分配给连续的库，故连续的线程最好访问连续的内存地址。

  ![image](https://tvm.apache.org/images/depthconv_tutorial/bank_conflicts.png)

###  Schedule Optimization

1. padding被申明为独立的一步，inline 以避免多余的内存分配

   ```python
   s = tvm.create_schedule(Output.op)
   s[PaddedInput].compute_inline()
   ```

2. 把channel分块

   ```python
   #创建cache，输入数据用shared memory
   IS = s.cache_read(PaddedInput, "shared", [DepthwiseConv2d])
   FS = s.cache_read(Filter, "shared", [DepthwiseConv2d])
   block_y = tvm.thread_axis("blockIdx.y")
   block_x = tvm.thread_axis("blockIdx.x")
   # bind the dimension of batch (N in NCHW) with block_y
   s[Output].bind(Output.op.axis[0], block_y)
   # bind the dimension of channel (C in NCHW) with block_x
   s[Output].bind(Output.op.axis[1], block_x)
   ```

   > 在GTX 1080上测试得：由于channel增加到64x64时，性能会下降，故我们修改schedule，将一个channel分成32x32的块，一个cuda负责一个32x32。
   >
   > ```python
   > blocking_h = 32
   > blocking_w = 32
   > # split the dimension of height (H in NCHW)
   > bx1, _ = s[Output].split(Output.op.axis[2], factor=blocking_h)
   > # split the dimension of width (W in NCHW)
   > bx2, _ = s[Output].split(Output.op.axis[3], factor=blocking_w)
   > # assign one 32 x 32 block to one cuda block
   > by = s[Output].fuse(Output.op.axis[0], Output.op.axis[1])
   > s[Output].bind(by, block_y)
   > bx = s[Output].fuse(bx1, bx2)
   > s[Output].bind(bx, block_x)
   > ```

3. 调整线程

   在cuda中安排32x32的线程

   ```python
   num_thread_y = 8
   num_thread_x = 8
   thread_y = tvm.thread_axis((0, num_thread_y), "threadIdx.y")
   thread_x = tvm.thread_axis((0, num_thread_x), "threadIdx.x")
   ty, yi = s[Output].split(h_dim, nparts=num_thread_y)
   tx, xi = s[Output].split(w_dim, nparts=num_thread_x)
   s[Output].reorder(ty, tx, yi, xi)
   s[Output].bind(ty, thread_y)
   s[Output].bind(tx, thread_x)
   ```

   这里的num_thread_xy都可以通过autotvm优化。

4. 引入virtual thread

   ```python
   num_vthread_y = 2
   num_vthread_x = 2
   num_thread_y = 8
   num_thread_x = 8
   thread_vy = tvm.thread_axis((0, num_vthread_y), "vthread", name="vy")
   thread_vx = tvm.thread_axis((0, num_vthread_x), "vthread", name="vx")
   thread_y = tvm.thread_axis((0, num_thread_y), "threadIdx.y")
   thread_x = tvm.thread_axis((0, num_thread_x), "threadIdx.x")
   # split the dimension of height (H in NCHW) twice
   tvy, vyi = s[Output].split(h_dim, nparts=num_vthread_y)
   ty, yi = s[Output].split(vyi, nparts=num_thread_y)
   # split the dimension of width (W in NCHW) twice
   tvx, vxi = s[Output].split(w_dim, nparts=num_vthread_x)
   tx, xi = s[Output].split(vxi, nparts=num_thread_x)
   # bind thread and vthread respectively
   s[Output].bind(tvy, thread_vy)
   s[Output].bind(tvx, thread_vx)
   s[Output].bind(ty, thread_y)
   s[Output].bind(tx, thread_x)
   s[Output].reorder(tvy, tvx, ty, tx, yi, xi)
   ```


## tuner实现

TVM设想一个动态生成代码的过程，上层只定义算子的数学含义，比如用函数式语言定义算法，称为Compute，底层的具体实现则针对不同的硬件自动完成

官方例子大致可以分为两部分：

* Compute 并在其上定义Tune Space

* 整体为template 用tuner去tune这个template

  ```python
  def get_network(name, batch_size):
      """Get the symbol definition and random weight of a network"""
      input_shape = (batch_size, 3, 224, 224)
      output_shape = (batch_size, 1000)
  
      if "resnet" in name:
          n_layer = int(name.split('-')[1])
          net, params = relay.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
      elif "vgg" in name:
          n_layer = int(name.split('-')[1])
          net, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
      elif name == 'mobilenet':
          net, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
      elif name == 'squeezenet_v1.1':
          net, params = relay.testing.squeezenet.get_workload(batch_size=batch_size, version='1.1', dtype=dtype)
      elif name == 'inception_v3':
          input_shape = (1, 3, 299, 299)
          net, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
      elif name == 'mxnet':
          # an example for mxnet model
          from mxnet.gluon.model_zoo.vision import get_model
          block = get_model('resnet18_v1', pretrained=True)
          net, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)
          net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
      else:
          raise ValueError("Unsupported network: " + name)
  
      return net, params, input_shape, output_shape
  ```

  得到template，这里直接得到了网络，如果要得到某一个算子，需要找到注册的算子如

  ```python
  NNVM_REGISTER_OP(conv2d)
  
  def get_operator(data_shape, out_channel, kernel_size, strides, padding, dtype="float32"):
      data = relay.var("data", shape=data_shape, dtype=dtype)
      body = layers.conv2d(data=data, channels=out_channel, kernel_size=kernel_size, strides=strides, padding=padding, name="conv2d")
      return relay.Function(relay.ir_pass.free_vars(body), body)
  
  def get_workload(batch_size, image_shape, out_channel, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), dtype="float32"):
      data_shape = (batch_size, *image_shape)
      op = get_operator(data_shape, out_channel, kernel_size, strides, padding, dtype=dtype)
      sym, params = create_workload(op)
      return sym, params, data_shape
  ```

在tunning之前，需要配置

```python
#### DEVICE CONFIG ####
target = tvm.target.cuda()

#### TUNING OPTION ####
network = 'resnet-18'
log_file = "%s.log" % network
dtype = 'float32'

tuning_option = {
    'log_filename': log_file,

    'tuner': 'xgb',
    'n_trial': 2000,
    'early_stopping': 600,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
    ),
}
```

tune的函数

```python
# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))
		# 如果要指明，在这里指定
        
        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=tsk_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)
                       ])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)
```

最后tune

```python
def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, input_shape, out_shape = get_network(network, batch_size=1)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params,
                                              ops=(relay.op.get("nn.conv2d"),))

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)

        # export library
        tmp = tempdir()
        filename = "net.tar"
        lib.export_library(tmp.relpath(filename))

        # load parameters
        ctx = tvm.context(str(target), 0)
        module = runtime.create(graph, lib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))

# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

# tune_and_evaluate(tuning_option)
```



