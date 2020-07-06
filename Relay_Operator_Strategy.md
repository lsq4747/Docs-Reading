# Relay Operator Strategy

每一个Relay算子需要注册

* 计算函数 a compute function
* 调度函数 a schedule function

但以上函数通常是针对每个目标而定的，而且对于同一个目标会有多种实现方法。于是引入算子策略Operator Strategy，对于每一个目标，选择算子最适合的实现。

[toc]

## OpStrategy

* 一个<kbd>OpStrategy</kbd>里有一系列的 <kbd>OpSpecialization</kbd>，每一个<kbd>OpSpecialization</kbd>中里有一系列的<kbd>OpImplementation</kbd>。

* <kbd>OpImplementation</kbd>是与<kbd>SpecializedCondition</kbd>相关联的。

* <kbd>SpecializedCondition</kbd>里是关于张量形状的条件，<u>它是由一系列子句组成，这些子句是由张量表达形式中的conjunctive normal form CNF形式定义的</u>。

  

## Implementation

算子策略的基本元素是<kbd>OpImplementation</kbd>，其中含有：

* 计算函数 a compute function
* 调度函数 a schedule function
* 名称 the name of the implementation
* 优先级 priority level

在编译过程中，在算子有多重表达时，Relay compile engine需要决定算子使用哪一个implementation。其中选择原理如下：

* 输入张量有确定形状时：
  1. 根据 **AutoTVM tuning logs**选择implementation。
  2. 如果没有AutoTVM模板，或所有AutoTVM模板都有<u>fallback configs</u>，那么选择**优先级priority level**最高的implementation。<u>但是这种情况下，优先级相同的implementation会出现问题，它们之中的任何一个都可能被选择</u>。
* 输入张量有符号形状时：只有**优先级priority level**最高的implementation可以。implementation完成后会进行更行。

在编译Relay模型前加上：

```python
logging.getLogger("compile_engine").setLevel(logging.INFO)
logging.getLogger("compile_engine").addHandler(logging.StreamHandler(sys.stdout))
```



## Strategy Function

策略函数，<kbd>FTVMStrategy</kbd>。

* 在给定一个工作负载时，策略函数决定应该使用哪一个计算函数和调度函数。
* 策略函数需要注册到每一个算子。
* 策略函数返回一个<kbd>OpStrategy</kbd>，给定了算子的属性、输入、输出类型、编译的目标。
* 编写策略函数一般用python

在<kbd>OpStrategy</kbd>中添加<kbd>OpImplementation</kbd>使用函数<kbd>add_implementation</kbd>:

```python
@tvm._ffi.register_object("relay.OpStrategy")
class OpStrategy(Object):
    """Operator strategy"""
    def __init__(self):
        self.__init_handle_by_constructor__(_make.OpStrategy)

    def add_implementation(self, compute, schedule, name="default", plevel=10):
        """Add an implementation to the strategy

        Parameters
        ----------
        compute : function (attrs: Attrs, inputs: List[Tensor], out_type: Type)
                           -> List[Tensor]
            The compute function.

        schedule : function (attrs: Attrs, outs: List[Tensor], target:Target) -> Schedule
            The schedule function.

        name : str
            The name of implementation.

        plevel : int
            The priority level of implementation.
        """
        _OpStrategyAddImplementation(self, compute, schedule, name, plevel)
```

以<kbd>python/tvm/relay/op/strategy/generic.py</kbd>中的一个函数为例：

```python
# conv2d_NCHWc
@override_native_generic_func("conv2d_NCHWc_strategy")
def conv2d_NCHWc_strategy(attrs, inputs, out_type, target):
    """conv2d_NCHWc generic strategy"""
    logger.warning("conv2d_NCHWc is not optimized for this platform.")
    strategy = _op.OpStrategy()
    if inputs[0].dtype == "int8" or inputs[0].dtype == "uint8":
        strategy.add_implementation(
            wrap_compute_conv2d(topi.nn.conv2d_NCHWc_int8, True, True),
            wrap_topi_schedule(topi.generic.schedule_conv2d_NCHWc_int8),
            name="conv2d_NCHWc_int8.generic")
    else:
        strategy.add_implementation(
            wrap_compute_conv2d(topi.nn.conv2d_NCHWc, True, True),
            wrap_topi_schedule(topi.generic.schedule_conv2d_NCHWc),
            name="conv2d_NCHWc.generic")
    return strategy
```

* 在<kbd>add_implementation</kbd>中，使用了两个封装函数来封装**计算函数**和**调度函数**：

  > * **计算函数**
  >
  >   ```python
  >   # conv2d
  >   def wrap_compute_conv2d(topi_compute, need_data_layout=False, need_out_layout=False,
  >                           has_groups=False):
  >       """Wrap conv2d topi compute"""
  >       def _compute_conv2d(attrs, inputs, out_type):
  >           padding = get_const_tuple(attrs.padding)
  >           strides = get_const_tuple(attrs.strides)
  >           dilation = get_const_tuple(attrs.dilation)
  >           data_layout = attrs.get_str("data_layout")
  >           out_layout = attrs.get_str("out_layout")
  >           out_dtype = attrs.out_dtype
  >           out_dtype = (inputs[0].dtype if out_dtype in ("same", "")
  >                        else out_dtype)
  >           args = [inputs[0], inputs[1], strides, padding, dilation]
  >           if has_groups:
  >               args.append(attrs.groups)
  >           if need_data_layout:
  >               args.append(data_layout)
  >           if need_out_layout:
  >               args.append(out_layout)
  >           args.append(out_dtype)
  >           return [topi_compute(*args)]
  >       return _compute_conv2d
  >   ```
  >
  >   以上封装函数中<kbd>_compute_conv2d</kbd>是满足<kbd>include/tvm/relay/op_attr_types.h</kbd>的<kbd>FTVMCompute</kbd>：
  >
  >   ```c++
  >   using FTVMCompute = runtime::TypedPackedFunc<
  >     Array<te::Tensor>(const Attrs& attrs,
  >                       const Array<te::Tensor>& inputs,
  >                       const Type& out_type)>;
  >   ```
  >
  > * **调度函数**
  >
  >   ```python
  >   def wrap_topi_schedule(topi_schedule):
  >       """Wrap TOPI schedule which doesn't use attrs"""
  >       def wrapper(attrs, outs, target):
  >           with target:
  >               return topi_schedule(outs)
  >       return wrapper
  >   ```
  >
  >   和compute function一样，封装完成的<kbd>wrapper</kbd>是满足<kbd>FTVMSchedule</kbd>的：
  >
  >   ```c++
  >   using FTVMSchedule = runtime::TypedPackedFunc<
  >     te::Schedule(const Attrs& attrs,
  >                  const Array<te::Tensor>& outs,
  >                  const Target& target)>;
  >   ```

* 在这个<kbd>conv2d</kbd>的策略函数中，根据条件不同，添加了两个<kbd>implementation</kbd>

* <u>以上的条件可以扩展到第三方库中</u>：

  > 如教程中所给出的例子，当cblas被包含在目标库中时添加这一个<kbd>implementation</kbd>：
  >
  > ```python
  > if "cblas" in target.libs:
  >     strategy.add_implementation(
  >         wrap_compute_dense(topi.x86.dense_cblas),
  >         wrap_topi_schedule(topi.x86.schedule_dense_cblas),
  >         name="dense_cblas.x86",
  >         plevel=15)
  > ```

* <u>可以添加针对特定范围内的形状的<kbd>implementation</kbd></u>。

  

## Register Strategy Function

### To An Operator

在定义了一个策略函数之后，可以用<kbd>python/tvm/relay/op/op.py</kbd>中的函数对算子进行注册：

```python
def register_strategy(op_name, fstrategy=None, level=10):
    """Register strategy function for an op.

    Parameters
    ----------
    op_name : str
        The name of the op.

    fstrategy : function (attrs: Attrs, inputs: List[Tensor], out_type: Type,
                          target:Target) -> OpStrategy
        The strategy function. Need to be native GenericFunc.

    level : int
        The priority level
    """
    if not isinstance(fstrategy, GenericFunc):
        assert hasattr(fstrategy, "generic_func_node")
        fstrategy = fstrategy.generic_func_node
    return register(op_name, "FTVMStrategy", fstrategy, level)
```

对于一些更加简单的算子，还有以下两种方法：

1. 对于**injective、broadcast、reduction**形式的算子，可以分别调用<kbd>register_injective_schedule</kbd>、<kbd>register_broadcast_schedule</kbd>、<kbd>register_reduce_schedule</kbd>来注册：

   ```python
   def register_injective_schedule(op_name, level=10):
       """Register injective schedule function for an op.
   
       Parameters
       ----------
       op_name : str
           The name of the op.
   
       level : int
           The priority level
       """
       return register_schedule(op_name, _schedule_injective, level)
   ```

   其中的<kbd>_schedule_reduce</kbd>等函数是经过以下封装的：

   ```python
   @generic_func
   def schedule_injective(attrs, outs, target):
       """Schedule injective ops"""
       with target:
           return topi.generic.schedule_injective(outs)
   
   _op._schedule_injective = schedule_injective
   ```

2. 对于其他算子，可以使用<kbd>register_schedule</kbd>来注册:

   ```python
   def register_schedule(op_name, schedule, level=10):
       """Register schedule function for an op.
   
       This is used when compute function is the same for all targets and only
       schedule is different. It requires FTVMCompute is already registered to
       the op.
   
       Parameters
       ----------
       op_name : str
           The name of the op.
   
       schedule : function (attrs: Attrs, outs: List[Tensor], target:Target) -> Schedule
           The schedule function. Need to be target.generic_func.
   
       level : int
           The priority level
       """
       fstrategy = _create_fstrategy_from_schedule(op_name, schedule)
       return register_strategy(op_name, fstrategy, level)
   ```

   其中的<kbd>schedule</kbd>是经过以下封装的，以pool为例：

   ```python 
   # pool_grad
   @generic_func
   def schedule_pool_grad(attrs, outs, target):
       """Schedule pooling gradient ops"""
       with target:
           return topi.generic.schedule_pool_grad(outs)
   ```
   
   可以发现，经过封装的函数都是满足<kbd>FTVMSchedule</kbd>的：
   
   ```c++
   using FTVMSchedule = runtime::TypedPackedFunc<
     te::Schedule(const Attrs& attrs,
                  const Array<te::Tensor>& outs,
                  const Target& target)>;
   ```
   
   

### For A New Target

当有一个新目标的时候，有两种方法注册策略：

1. 在<kbd>python/tvm/relay/op/strategy</kbd>中添加一个新的目标文件，然后使用已经实现的策略。
2. 也可以在python库之外,，如<kbd>vta/python/vta/top/op.py</kbd>，给新目标注册策略。