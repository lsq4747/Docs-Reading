# Adding an Operator to Relay

[toc]

## Steps

*  注册运算符算子、类型信息

* 定义函数，产生call node，注册一个python API hook

* 封装python API hook

  

## Registering an operator

Relay需要一些附加类型的信息才可以使用TVM的operator  registry。

算子的类型由**输入类型与输出类型之间的relation**来决定。我们可以将**输入类型与输出类型之间的relation**看做一个函数，它输入任何一项都有可能缺失的input types and output types ，返回是一个补齐的、满足这种relation的input types and output types。确保输入、输出的类型正确。

<kbd>BroadcastRel</kbd>用来检测有两输入、一输出的算子：

```c++
bool BroadcastRel(const Array<Type>& types,
                  int num_inputs,
                  const Attrs& attrs,
                  const TypeReporter& reporter);
```

<kbd>add</kbd>即两输入、一输出的算子，用到了<kbd>BroadcastRel</kbd>去确保输入输出类型的正确，用<kbd>RELAY_REGISTER_OP</kbd>注册算子<kbd>add</kbd>如下：

```c++
RELAY_REGISTER_OP("add")
    .set_num_inputs(2)//变量数目
    .add_argument("lhs", "Tensor", "The left hand side tensor.")
    .add_argument("rhs", "Tensor", "The right hand side tensor.")//变量名、说明
    .set_support_level(1)//数字越大表示积分或外部支持的算子越少
    .add_type_rel("Broadcast", BroadcastRel);
```

文档中描述，在<kbd>src/relay/op/tensor/binary.cc</kbd>里使用了<kbd>RELAY_REGISTER_OP</kbd>注册了算子。不知是版本原因还是其他原因，这里并没有使用<kbd>RELAY_REGISTER_OP</kbd>去注册算子，而是将算子分类之后，不同的类别使用不同的宏去注册。但是在<kbd>src/relay/op/op_common.h</kbd>中，每种分类都写对应的了<kbd>RELAY_REGISTER_OP</kbd>：

```c++
#define RELAY_REGISTER_BINARY_OP(OpName)                             \
  TVM_REGISTER_GLOBAL("relay.op._make." OpName)                      \
  .set_body_typed([](Expr lhs, Expr rhs) {                           \
    static const Op& op = Op::Get(OpName);                           \
    return CallNode::make(op, {lhs, rhs}, Attrs(), {});              \
  });                                                                \
  RELAY_REGISTER_OP(OpName)                                          \
  .set_num_inputs(2)                                                 \
  .add_argument("lhs", "Tensor", "The left hand side tensor.")       \
  .add_argument("rhs", "Tensor", "The right hand side tensor.")      \
  .add_type_rel("Broadcast", BroadcastRel)                           \
  .set_attr<TOpPattern>("TOpPattern", kBroadcast)                    \
  .set_attr<TOpIsStateful>("TOpIsStateful", false)                   \
  .set_attr<FInferCorrectLayout>("FInferCorrectLayout",              \
                                 BinaryBroadcastLayout)
```

以下为<kbd>src/relay/op/tensor/binary.cc</kbd>中的几种算子：

1. <kbd>RELAY_REGISTER_BINARY_OP</kbd>

   包含 加、减、左右移、乘、除、mod、乘方、逻辑操作、按位操作等等。

   ```c++
   // Addition
   RELAY_REGISTER_BINARY_OP("add")
   .describe("Elementwise add with with broadcasting")
   .set_support_level(1)
   .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::add));
   ```

2. <kbd>RELAY_REGISTER_CMP_OP</kbd>

   包含 等于、不等于、大于、大于等于、小于、小于等于、远大于、远大于等于等等。

   ```c++
   RELAY_REGISTER_CMP_OP("equal")
   .describe("Elementwise equal compare with broadcasting")
   .set_support_level(4)
   .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::equal));
   ```

于此同时，还有很多<kbd>.cc</kbd>注册了大量的算子，有一些使用了<kbd>RELAY_REGISTER_OP</kbd>，有一些使用了别的宏，如<kbd>src/relay/op/tensor/reduce.cc</kbd>中的<kbd>RELAY_REGISTER_REDUCE_OP</kbd>,<kbd>src/relay/op/tensor/unary.cc</kbd>中的<kbd>RELAY_REGISTER_UNARY_OP</kbd>等等。



## Create a call node

这一步为一个函数，输入为算子的变量，输出为一个call node。

```c++
TVM_REGISTER_GLOBAL("relay.op._make.add")
    .set_body_typed<Expr(Expr, Expr)>([](Expr lhs, Expr rhs) {	//算子的变量
        static const Op& op = Op::Get("add");
      return CallNode::make(op, {lhs, rhs}, Attrs(), {});	//返回一个call node
    });
```



## Python API hook Wrap

在Relay中，一般通过上一步<kbd>TVM_REGISTER_GLOBAL</kbd>输出的函数应该被封装在一个单独的python函数中。如果函数中调用了别的算子，将它们合并在一起会更方便。

在<kbd>python/tvm/relay/op/tensor.py </kbd>中有张量运算的算子，以<kbd>add</kbd>为例：

```python
def add(lhs, rhs):
    """Addition with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.

    Examples
    --------
    .. code:: python

      x = relay.Var("a") # shape is [2, 3]
      y = relay.Var("b") # shape is [2, 1]
      z = relay.add(x, y)  # result shape is [2, 3]
    """
    return _make.add(lhs, rhs)
```

也有函数经过封装之后，调用的接口更简单,如<kbd>concatenate</kbd>：

```python
def concatenate(data, axis):
    """Concatenate the input tensors along the given axis.

    Parameters
    ----------
    data : Union(List[relay.Expr], Tuple[relay.Expr])
        A list of tensors.
    axis : int
        The axis along which the tensors are concatenated.

    Returns
    -------
    result: relay.Expr
        The concatenated tensor.
    """
    data = list(data)
    if not data:
        raise ValueError("relay.concatenate requires data to be non-empty.")
    if not isinstance(axis, int):
        raise ValueError("For now, we only support integer axis")
    return _make.concatenate(Tuple(data), axis)
```



## Gradient Operators

文档里还讲了梯度运算，这里先以<kbd>log</kbd>为例，其中<kbd>ones_like(x) / x</kbd>为梯度的计算式，而前面的<kbd>grad * </kbd>是为了进行梯度的累积计算：

```python
@register_gradient("log")
def log_grad(orig, grad):
    """Returns [grad * (1 / x)]"""
    x = orig.args[0]
    return [grad * ones_like(x) / x]
```

> 这里穿插补充一个函数<kbd>collapse_sum_like</kbd>，它将<kbd>data</kbd>折叠为<kbd>collapse_type</kbd>的类型，以确保形状是匹配的：
>
> ```c++
> def collapse_sum_like(data, collapse_type):
>     """Return a scalar value array with the same shape and type as the input array.
> 
>     Parameters
>     ----------
>     data : relay.Expr
>         The input tensor.
> 
>     collapse_type : relay.Expr
>         Provide the type to collapse to.
> 
>     Returns
>     -------
>     result : relay.Expr
>         The resulting tensor.
>     """
>     return _make.collapse_sum_like(data, collapse_type)
> ```

<kbd>multiply</kbd>算子中就使用了<kbd>collapse_sum_like</kbd>：

```python
@register_gradient("multiply")
def multiply_grad(orig, grad):
    """Returns [grad * y, grad * x]"""
    x, y = orig.args
    return [collapse_sum_like(grad * y, x),
            collapse_sum_like(grad * x, y)]
```

除此之外，还需要在C++添加，写法和python中的类似，需要先include<kbd>src/relay/pass/pattern_util.h</kbd>，在Relay AST中创建node需要用到里面的函数：

```c++
tvm::Array<Expr> MultiplyGrad(const Expr& orig_call, const Expr& output_grad) {
    //c++中不能像python中一样重载算子，所以需要Downcast
    const Call& call = orig_call.Downcast<Call>();
    return { CollapseSumLike(Multiply(output_grad, call.args[1]), call.args[0]),
             CollapseSumLike(Multiply(output_grad, call.args[0]), call.args[1]) };
}
```

梯度算子的注册有一些不同，需要在基础算子注册结束处加上一个<kbd>set_attr</kbd>：

```c++
RELAY_REGISTER_OP("multiply")
    // ...
    // Set other attributes
    // ...
    .set_attr<FPrimalGradient>("FPrimalGradient", MultiplyGrad);//为了梯度计算加的
```





