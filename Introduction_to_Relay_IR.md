# Introduction to Relay IR

[toc]

## Goals of Relay

* data flow表示

* let-binding表示 

* 以上两种混合

  

## Dataflow graph 

计算图*computational graph*——数据流图***data flow graph***

是一种有向无环图

* 每一个data flow node 都是一个CallNode
* Relay Python DSL 可快速构造 data flow  graph
* 打印Relay程序时，我们每行打印一个CallNode，并为每个CallNode分配一个临时id（%1，2，3…），则每个节点都可以引用。（shared node问题）
* 支持多个函数，可调用、可递归

```python

def @muladd(%x, %y, %z) {
  %1 = mul(%x, %y)
  %2 = add(%1, %z)
  %2
}
def @myfunc(%x) {
  %1 = @muladd(%x, 1, 2)
  %2 = @muladd(%1, 2, 3)
  %2
}
```



## Let Binding

是一个data structure <kbd>Let(var, value, body)</kbd>

将value赋值给var，返回结果到body

* 一系列let binding构造的程序 与 dataflow程序 逻辑上等同

  > A-normal form是let binding的嵌套形式

![dataflow&let](https://raw.githubusercontent.com/tvmai/tvmai.github.io/master/images/relay/dataflow_vs_func.png)

* Dataflow与A-normal Form的python code和text form基本相同，但是**AST structure**不同。

  > 图上两种不同的结构会影响我们要写的compiler code:
  >
  > * dataflow 中可以直接从add访问 log
  > * A-normal form中需要一个从 variable到 bound value的map

* let binding指定了计算的scope。因为计算在let node进行。

  > dataflow不指定计算的scope，方便，优化初期
  >
  > let binding，在实际运行时，需要精确计算的scope，后期优化

  



