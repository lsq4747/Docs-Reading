# TVM Runtime System

对于TVM runtime，我们需要满足如下的需求：

* **Deployment** 从python/JavaScript/C++中调用编译后的函数
* **Debug**  在python中定义一个函数，并在编译后的函数中调用
* **Link** 编写驱动代码调用CUDA，并从编译后的<u>host函数</u>中调用
* **Prototype** 从python中定义一个IR pass，在c++调用
* **Expose** <u>将C++中的 Compiler Stack给前端</u>
* **Experiment** 将编译好的函数放到嵌入式设备上运行

我们希望能从任何语言定义函数，并从另一个语言调用，并且runtime core尽可能小，以便部署在嵌入式设备上。

[toc]

## PackedFunc

<kbd>PackedFunc</kbd>是一种解决方法，一个c++例子如下：

```c++
#include <tvm/runtime/packed_func.h>

void MyAdd(TVMArgs args, TVMRetValue* rv) {
  // automatically convert arguments to desired type.
  int a = args[0];
  int b = args[1];
  // automatically assign value return to rv
  *rv = a + b;
}

void CallPacked() {
  PackedFunc myadd = PackedFunc(MyAdd);
  // get back 3
  int c = myadd(1, 2);
}
```

我们可以从动态语言（如python）调用<kbd>PackedFunc</kbd>，以下是注册并从python中调用的例子：

```c++
// register a global packed function in c++
TVM_REGISTER_GLOBAL("myadd")
.set_body(MyAdd);
```

```python
import tvm

myadd = tvm.get_global_func("myadd")
# prints 3
print(myadd(1, 2))
```

<kbd>PackedFunc</kbd>中的TVMArgs和TVMRetValue限制了一些可传递的类型，以下是可传递的类型：

* int float string
* PackedFunc
* Module 编译模块
* DLTensor* 张量对象交换
* <u>TVM Object to represent any object in IR</u>

由于<kbd>PackedFunc</kbd>可以接受另一个<kbd>PackedFunc</kbd>作为参数，我们可以将python中的函数作为<kbd>PackedFunc</kbd>传递到C++中：

```c++
TVM_REGISTER_GLOBAL("callhello")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  PackedFunc f = args[0];
  f("hello world");
});
```

```python
import tvm

def callback(msg):
  print(msg)

# convert to PackedFunc
f = tvm.convert(callback)
callhello = tvm.get_global_func("callhello")
# prints hello world
callhello(f)
```

<kbd>PackedFunc</kbd>同时用于compiler和deployment stack。TVM所有的compiler pass函数都以PackedFunc形式给到前端；编译后的函数以PackedFunc形式返回。

为了使runtime小，我们将<u>IR Object support</u>从runtime里面分离出来，这样的runtime大约为200k-600k。

## Module

由于TVM支持多种设备，则需要支持不同类型的设备的驱动程序。

TVM将编译后的对象定义为Module，**我们可以以PackedFunc形式从Module中获得编译后的函数，就可以用它来连接设备代码和PackedFunc**。

同时，<kbd>ModuleNode</kbd>是一个抽象类，可以支持CUDA、Metal、OpenCL等各种模块。

## Remote Deployment

通过PackedFunc和Module，可以将函数直接运到远程设备中。有一个**RPCModule**将参数序列化做数据移动，并远程启动，而且RPC的服务器小，可以捆绑在runtime。于是我们可以在Android/树莓派/iPhone/浏览器上启动TVM RPC服务器。

For example, to test the correctness of generated code on iPhone, we no longer have to write test-cases in swift/objective-c from scratch – We can use RPC to execute on iPhone, copy the result back and do verification on the host via numpy. 

## TVM Object and Compiler Stack

引入了<kbd>Object</kbd>这个类为了

* 能序列化任何语言和IR
* 能在前端语言操作IR

编译器栈中的语言对象都是<kbd>Object</kbd>的子类，用type_key来标识，shared_ptr来跟踪引用。同时，使用<kbd>ObjectRef</kbd>这个类来表示对<kbd>Object</kbd>的引用。<kbd>Object</kbd>每个子类都需要定义一个<kbd>VisitAttr</kbd>函数：

```c++
class AttrVisitor {
public:
  virtual void Visit(const char* key, double* value) = 0;
  virtual void Visit(const char* key, int64_t* value) = 0;
  virtual void Visit(const char* key, uint64_t* value) = 0;
  virtual void Visit(const char* key, int* value) = 0;
  virtual void Visit(const char* key, bool* value) = 0;
  virtual void Visit(const char* key, std::string* value) = 0;
  virtual void Visit(const char* key, void** value) = 0;
  virtual void Visit(const char* key, Type* value) = 0;
  virtual void Visit(const char* key, ObjectRef* value) = 0;
  // ...
};

class BaseAttrsNode : public Object {
public:
  virtual void VisitAttrs(AttrVisitor* v) {}
  // ...
```

如TensorNode：

```c++
class TensorNode : public Object {
 public:
  /*! \brief The shape of the tensor */
  Array<PrimExpr> shape;
  /*! \brief data type in the content of the tensor */
  DataType dtype;
  /*! \brief the source operation, can be None */
  Operation op;
  /*! \brief the output index from source operation */
  int value_index{0};
  /*! \brief constructor */
  TensorNode() {}

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("shape", &shape);
    v->Visit("dtype", &dtype);
    v->Visit("op", &op);
    v->Visit("value_index", &value_index);
  }
  TVM_DLL static Tensor make(Array<PrimExpr> shape,
                             DataType dtype,
                             Operation op,
                             int value_index);

  static constexpr const char* _type_key = "Tensor";
  TVM_DECLARE_FINAL_OBJECT_INFO(TensorNode, Object);
};
```

在这个例子中Operation和Array<PrimExpr>都是<kbd>ObjectRef</kbd>。我们可以通过这个函数，访问节点、递归地序列化语言对象、在前端语言中获得对象的成员。