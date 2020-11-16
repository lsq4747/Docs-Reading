# Bring Your Own Codegen To TVM

硬件后端应：

* 提供MKLDNN或cuDNN等库，并提供许多常用的深度学习运算符
* 提供TensorRT等框架，让用户可以以某种形式描述模型

本文档讲了如何实现自己的Codegen，并将其注册为Relay后端编译器。

1. 生成C code

   如果硬件端有比较好的C/C++库，由于C source模块与到TVM runtime模块是完全兼容的，则生成的代码可以被C/C++编译器编译。故需要一个为子图生成C code的codegen和一个C source模块集成到TVM runtime模块

2. 生成graph表示

   此时，不仅需要代码生成器，还需要一个自定义的TVM runtime模块，让它可以知道如何执行这个图。

[toc]

## Implement a C Codegen

这里的目标是用已实现的operator functions去生成C code，这里的codegen不依赖第三方库，如下，使用了两个宏定义定义了1D和2D的二进制算子：

```c++
    // Append some common macro for operator definition.
    const char* operator_macro = R"op_macro(
    #define CSOURCE_BINARY_OP_1D(p_ID_, p_OP_, p_DIM1_, p_DTYPE)       \
      extern "C" void p_ID_(p_DTYPE* a, p_DTYPE* b, p_DTYPE* out) {    \
        for (int64_t i = 0; i < p_DIM1_; ++i) {                        \
          out[i] = a[i] p_OP_ b[i];                                    \
        }                                                              \
      }

    #define CSOURCE_BINARY_OP_2D(p_ID_, p_OP_, p_DIM1_, p_DIM2_, p_DTYPE)  \
      extern "C" void p_ID_(p_DTYPE* a, p_DTYPE* b, p_DTYPE* out) {        \
        for (int64_t i = 0; i < p_DIM1_; ++i) {                            \
          for (int64_t j = 0; j < p_DIM2_; ++j) {                          \
            int64_t k = i * p_DIM2_ + j;                                   \
            out[k] = a[k] p_OP_ b[k];                                      \
          }                                                                \
        }                                                                  \
      }
    )op_macro";
```

用这两个宏定义这样就可以进行1D2D张量的计算，以下面的子图为例：

```c++
c_compiler_input0
       |
      add <-- c_compiler_input1
       |
    subtract <-- c_compiler_input2
       |
    multiply <-- c_compiler_input3
       |
      out
```

那么最终的目标是生成如下的代码：

```c++
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <dlpack/dlpack.h>
#include <cstdint>
#include <cstring>
#include <iostream>

#define GCC_BINARY_OP_1D(p_ID_, p_OP_, p_DIM1_)           \
  extern "C" void p_ID_(float* a, float* b, float* out) { \
    for (int64_t i = 0; i < p_DIM1_; ++i) {               \
      out[i] = a[i] p_OP_ b[i];                           \
    }                                                     \
  }

#define GCC_BINARY_OP_2D(p_ID_, p_OP_, p_DIM1_, p_DIM2_)  \
  extern "C" void p_ID_(float* a, float* b, float* out) { \
    for (int64_t i = 0; i < p_DIM1_; ++i) {               \
      for (int64_t j = 0; j < p_DIM2_; ++j) {             \
        int64_t k = i * p_DIM2_ + j;                      \
        out[k] = a[k] p_OP_ b[k];                         \
      }                                                   \
    }                                                     \
  }

// 图中三个节点的函数实现
GCC_BINARY_OP_2D(gcc_0_0, *, 10, 10);
GCC_BINARY_OP_2D(gcc_0_1, -, 10, 10);
GCC_BINARY_OP_2D(gcc_0_2, +, 10, 10);

// 分配缓冲区、调用相应函数
extern "C" void gcc_0_(float* gcc_input0, float* gcc_input1,
                       float* gcc_input2, float* gcc_input3, float* out) {
  float* buf_0 = (float*)malloc(4 * 100);
  float* buf_1 = (float*)malloc(4 * 100);
  gcc_0_2(gcc_input0, gcc_input1, buf_0);
  gcc_0_1(buf_0, gcc_input2, buf_1);
  gcc_0_0(buf_1, gcc_input3, out);
  free(buf_0);
  free(buf_1);
}

// 封装函数，与TVM runtime兼容
extern "C" int gcc_0_wrapper(DLTensor* arg0, DLTensor* arg1, DLTensor* arg2,
                             DLTensor* arg3, DLTensor* out) {
  gcc_0_(static_cast<float*>(arg0->data), static_cast<float*>(arg1->data),
         static_cast<float*>(arg2->data), static_cast<float*>(arg3->data),
         static_cast<float*>(out->data));
  return 0;
}
//TVM宏，将所有张量都打包到gcc_0中，课直接调用执行子图
TVM_DLL_EXPORT_TYPED_FUNC(gcc_0, gcc_0_wrapper);
```

之后文档的其余部分，将逐步实现一个codegen来生成上述代码，codegen必须位于<kbd>src/relay/backend/contrib/</kbd>

```c++
                     subgraph                                subgraph
TVM backend -----------------------------> CSourceCodegen -------------> CodegenC
       ^                                       |    ^                       |
       |                                       |    |                       |
       ----------------------------------------      ------------------------
          generated C source runtime module              generated C code
```

如上图所示，TVM发现一个子图被注册的编译器标签(<kbd>ccomplier</kbd>)注释时，TVM调用<kbd>CSourceCodegen</kbd>并传递子图。<kbd>CSourceCodegen</kbd>中的<kbd>CreateCSourceModule</kbd>将生成子图的C代码，并将其封装成C source runtime module，给TVM后端运行。

这其中有两个类：**CodegenC**、**CSourceCodegen**

### Implement CodegenC

首先，在<kbd>codegen.cc</kbd>中，<kbd>tvm.relay.contrib</kbd>的namespace下建立第一个codegen的类

```c++
namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

/*!
 * \brief An example codegen that is only used for quick prototyping and testing
 * purpose. Only several binary options are covered. Users
 * may need to extend them to cover more operators.
 */
class CodegenC : public MemoizedExprTranslator<std::vector<Output>>, public CodegenCBase {
 public:
  explicit CodegenC(const std::string& id) { this->ext_func_id_ = id; }

  std::vector<Output> VisitExprDefault_(const Object* op) 

  std::vector<Output> VisitExpr_(const VarNode* node) 

  std::vector<Output> VisitExpr_(const ConstantNode* cn) 
      
  std::vector<Output> VisitExpr_(const CallNode* call) 

  std::string JIT(const std::vector<Output>& out) 

 private:
  /*! \brief The function id that represents a C source function. */
  std::string ext_func_id_ = "";
  /*! \brief The index of a wrapped C function. */
  int func_idx = 0;
  /*! \brief The index of allocated buffers. */
  int buf_idx_ = 0;
  /*! \brief The index of global constants. */
  int const_idx_ = 0;
  /*! \brief The arguments of a C compiler compatible function. */
  Array<Var> ext_func_args_;
  /*! \brief The statements of a C compiler compatible function. */
  std::vector<std::string> ext_func_body;
  /*! \brief The declaration statements of a C compiler compatible function. */
  std::vector<std::string> func_decl_;
  /*! \brief The declaration statements of buffers. */
  std::vector<std::string> buf_decl_;
};
```

<kbd>CodegenC</kbd>继承了两类，一个是<kbd>MemoizedExprTranslator</kbd>，是一个<kbd>ExprFunctor</kbd>的封装器，可以生成子图函数；另一个是<kbd>CodegenCBase</kbd>，提供了生成封装函数的能力与具体程序。

#### Code Generation for Operators

首先，对于<kbd>std::vector<Output> VisitExpr_(const CallNode* call)</kbd>。这个函数在遍历子图的时候，访问了所有调用节点，每一个节点都包含了一个我们需要落实到硬件端的算子。因此，我们需要按照拓扑顺序生成相应算子的C code。这个函数的实现步骤如下：

1. **Generate the function declaration**

   ```c++
       std::ostringstream macro_stream;
       std::ostringstream decl_stream;
       std::ostringstream buf_stream;
   
       std::string func_name = ext_func_id_ + "_" + std::to_string(func_idx++);
   
       // Make function declaration
       macro_stream << "CSOURCE_BINARY_OP_" << call->args.size() << "D(" << func_name << ", ";
   
       if (IsOp(call, "add")) {
         macro_stream << "+";
       } else if (IsOp(call, "subtract")) {
         macro_stream << "-";
       } else if (IsOp(call, "multiply")) {
         macro_stream << "*";
       } else {
         LOG(FATAL) << "Unrecognized op";
       }
   
       auto in_shape = GetShape(call->args[0]->checked_type());
       for (size_t i = 0; i < in_shape.size(); ++i) {
         macro_stream << ", " << in_shape[i];
       }
   
       const auto* type_node = call->checked_type().as<TensorTypeNode>();
       CHECK(type_node);
       const auto& dtype = GetDtypeString(type_node);
       macro_stream << ", " << dtype;
   
       macro_stream << ");";
       func_decl_.push_back(macro_stream.str());
   ```

   我们可以从CallNode中获得函数名、算子种类、张量形状这些信息，生成类似的代码：

   ```c++
   GCC_BINARY_OP_2D(gcc_0_0, *, 10, 10);
   GCC_BINARY_OP_2D(gcc_0_1, -, 10, 10);
   GCC_BINARY_OP_2D(gcc_0_2, +, 10, 10);
   ```

   我们将生成的代码给到<kbd>func_decl_</kbd>，这说明我们已经完成了对子图的遍历，获得了函数的声明。

2. **Generate the function call**

   之后，我们要生成一个具有正确的输入输出的函数调用，我们需要访问它的参数才可以知道调用的时候需要哪些输入。

   ```c++
   // Make function call when visiting arguments
       bool first = true;
       decl_stream << func_name << "(";
       for (size_t i = 0; i < call->args.size(); ++i) {
         //一个递归调用，用来访问当前函数的参数，一个参数可以是另一个节点的输出  
         auto res = VisitExpr(call->args[i]);
         for (auto out : res) {
           if (!first) {
             decl_stream << ", ";
           }
           first = false;
           decl_stream << out.name;
         }
       }
   ```

   这一步中没有关闭函数的后括号，因为没有将最后一个参数（输出）放进去。

   最后生成的代码类型如下：

   ```c++
   gcc_0_0(buf_1, gcc_input3, out);
   ```

3. **Generate the output buffer**

   我们需要缓冲区来储存中间结果，生成缓冲区需要知道缓冲区的类型和大小

   ```c++
       std::string out = "buf_" + std::to_string(buf_idx_++);
       auto out_shape = GetShape(call->checked_type());
       int out_size = 1;
       for (size_t i = 0; i < out_shape.size(); ++i) {
         out_size *= out_shape[i];
       }
       buf_stream << dtype << "* " << out <<
         " = (" << dtype << "*)std::malloc(4 * " << out_size << ");";
       buf_decl_.push_back(buf_stream.str());
   
       decl_stream << ", " << out << ");";
       ext_func_body.push_back(decl_stream.str());
   ```

   在分配完输出缓冲区之后，就可以后括号关闭函数，并将生成的函数给到<kbd>ext_func_body</kbd>中。

   最后生成的代码类型如下：

   ```c++
    float* buf_0 = (float*)malloc(4 * 100);
    float* buf_1 = (float*)malloc(4 * 100);
   ```

4. **Update output buffer**

   最后，为了让下一个节点知道它应该接受哪一个缓冲区，我们需要更新变量<kbd>output</kbd>:

   ```c++
       // Update output buffer
       // Note C codegen only handles TensorType. Therefore, we don't flatten
       // tuples and only return a single vaule.
       Output output;
       output.name = out;
       output.dtype = dtype;
       output.need_copy = true;
       output.size = out_size;
       return {output};
   ```

#### Code Generation for Input Variables

对于<kbd>VarNode</kbd>，它表示模型中输入的张量，它的重要信息是<kbd>name hint</kbd>，如data、weight等。我们只需要更新变量<kbd>output</kbd>来传递<kbd>name hint</kbd>。

```c++
  std::vector<Output> VisitExpr_(const VarNode* node) final {
    ext_func_args_.push_back(GetRef<Var>(node));
    Output output;
    output.name = node->name_hint();
    return {output};
  }
```

#### Code Emitting

除此之外，这个类里还有一个JIT函数。我们调用<kbd>JitImpl</kbd>，将之前生成的子图函数封装，给TVM runtime调用。

文档里给出了<kbd>JitImpl</kbd>的调用例子：

```c++
JitImpl("gcc_0" /* Subgraph symbol (ID) */,
        {"gcc_input0", "gcc_input1", "gcc_input2", "gcc_input3"} /* Input arguments */,
        {"float *buf_0 = (float*)malloc(4 * 20)", ...} /* Buffer allocations */,
        {"gcc_0_2(gcc_input0, gcc_input1, buf_0);"} /* Function body */,
        {"out"} /* Output */);
```

这样会生成三个函数：

* 子图函数<kbd>gcc_0\_</kbd>，包含生成的代码
* 封装函数<kbd>gcc_0__wrapper\_</kbd>，将数据投射到正确的类型，再调用<kbd>gcc_0\_</kbd>
* <kbd>gcc_0</kbd>，TVM runtime兼容，它解包TVM打包的张量，再调用封装函数

```c++
  /*!
   * \brief Emit the source code that invokes C compiler compatible wrappers.
   *
   * \return The emitted code.
   */
  std::string JIT(const std::vector<Output>& out) {
    // Write function macros
    for (auto decl : func_decl_) {
      code_stream_ << decl << "\n";
    }
    return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body, out);
  }
```

### Implement CSourceCodegen

在<kbd>codegen.cc</kbd>中，建立第二个类<kbd> CSourceCodegen</kbd>。

```c++
class CSourceCodegen : public CSourceModuleCodegenBase {
 public:
  void GenCFunc(const Function& func)

  runtime::Module CreateCSourceModule(const ObjectRef& ref) 

 private:
  std::ostringstream code_stream_;
};

```

#### Implement GenCFunc

<kbd>GenCFunc</kbd>使用<kbd>CodegenC</kbd>去遍历一个函数，然后获得生成的C code。

```c++
 void GenCFunc(const Function& func) {
    CHECK(func.defined()) << "Input error: expect a Relay function.";

    // Record the external symbol for runtime lookup.
    auto sid = GetExtSymbol(func);

    CodegenC builder(sid);
    auto out = builder.VisitExpr(func->body);
    code_stream_ << builder.JIT(out);
  }
```

#### Implement CreateCSourceModule

整函数为外库建立了一个runtime module，这里建立的<kbd>CSourceModule</kbd>可以直接被编译，且与<u>TVM生成的DSOModule连接在一起</u>。

```c++
runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
    // Create headers
    code_stream_ << "#include <cstring>\n";
    code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
    code_stream_ << "#include <tvm/runtime/packed_func.h>\n";
    code_stream_ << "#include <dlpack/dlpack.h>\n";

    // Append some common macro for operator definition.
    const char* operator_macro = R"op_macro(
    #define CSOURCE_BINARY_OP_1D(p_ID_, p_OP_, p_DIM1_, p_DTYPE)       \
      extern "C" void p_ID_(p_DTYPE* a, p_DTYPE* b, p_DTYPE* out) {    \
        for (int64_t i = 0; i < p_DIM1_; ++i) {                        \
          out[i] = a[i] p_OP_ b[i];                                    \
        }                                                              \
      }

    #define CSOURCE_BINARY_OP_2D(p_ID_, p_OP_, p_DIM1_, p_DIM2_, p_DTYPE)  \
      extern "C" void p_ID_(p_DTYPE* a, p_DTYPE* b, p_DTYPE* out) {        \
        for (int64_t i = 0; i < p_DIM1_; ++i) {                            \
          for (int64_t j = 0; j < p_DIM2_; ++j) {                          \
            int64_t k = i * p_DIM2_ + j;                                   \
            out[k] = a[k] p_OP_ b[k];                                      \
          }                                                                \
        }                                                                  \
      }
    )op_macro";

    code_stream_ << operator_macro << "\n\n";

    if (ref->IsInstance<FunctionNode>()) {
      GenCFunc(Downcast<Function>(ref));
    } else if (ref->IsInstance<IRModuleNode>()) {
      IRModule mod = Downcast<IRModule>(ref);
      for (const auto& it : mod->functions) {
        GenCFunc(Downcast<Function>(it.second));
      }
    } else {
      LOG(FATAL) << "The input ref is expected to be a Relay function or module"
                 << "\n";
    }

    // Create a CSourceModule
    const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
    CHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";
    return (*pf)(code_stream_.str(), "cc");
  }
```

### Register Your Codegen

最后，将codegen注册到TVM后端

```c++
runtime::Module CCompiler(const ObjectRef& ref) {
  CSourceCodegen csource;
  return csource.CreateCSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.ccompiler").set_body_typed(CCompiler);
```

其中，<kbd>CCompiler</kbd>就是那个标注。

## Implement a Codegen for Your Representation

除了实现一个C codegen之外，硬件端可能需要其他形式的图，如JSON。这时，我们可以修改已经实现的<kbd>CodegenC</kbd>类来生成图，并自定义一个runtime module，让TVM runtime知道如何去执行这个图。

文档中举例了一种图的表示法，命名为<kbd>ExampleJSON</kbd>。

子图：

```c++
 input0
   |
  add <-- input1
   |
subtract <-- input2
   |
multiply <-- input3
   |
  out
```

ExampleJSON表示：

```c++
subgraph_0
  input 0 10 10
  input 1 10 10
  input 2 10 10
  input 3 10 10
  add 4 inputs: 0 1 shape: 10 10
  sub 5 inputs: 4 2 shape: 10 10
  mul 6 inputs: 5 3 shape: 10 10
```

我们最终的目标是实现如下TVM runtime module来执行这个图：

```c++
runtime::Module ExampleJsonCompiler(const NodeRef& ref) {
    ExampleJsonCodeGen codegen(ref);
    std::string code = codegen.gen(); // codegen通过取一个子图来生成ExampleJSON的代码
    const auto* pf = runtime::Registry::Get("module.examplejson_module_create"); //获得一个自定义runtime module的函数指针
    CHECK(pf != nullptr) << "Cannot find ExampleJson module to create the external runtime module";
    return (*pf)(code);
}
TVM_REGISTER_GLOBAL("relay.ext.examplejsoncompiler").set_body_typed(ExampleJsonCompiler);
```

### Implement ExampleJsonCodeGen

与<kbd>CodegenC</kbd>类似，继承了<kbd>MemoizedExprTranslator</kbd>，是一个<kbd>ExprFunctor</kbd>的封装器，可以生成子图函数；但没有继承<kbd>CodegenCBase</kbd>，因为不需要TVM C++封装器。文档中因为版本问题，继承的是<kbd>ExprVisitor</kbd>。

```c++
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>

#include <fstream>
#include <sstream>

namespace tvm {
namespace relay {
namespace contrib {

class ExampleJsonCodeGen : public ExprVisitor {
  public:
    explicit ExampleJsonCodeGen();

    // 得到图的代码
    void VisitExpr_(const VarNode* node) { /* Skip in this example. */ }
    void VisitExpr_(const CallNode* call) final { /* Skip in this example. */ }

    // 定义了一个内部API gen，来获取一个子图并生成一个ExampleJSON代码
    std::string gen(NodeRef& ref) {
        this->code = "";
        if (ref->IsInstance<FunctionNode>()) {
            this->visit(Downcast<Function>(ref));
        } else if (ref->IsInstance<relay::ModuleNode>()) {
            relay::Module mod = Downcast<relay::Module>(ref);
            for (const auto& it : mod->functions) {
                this->visit(Downcast<Function>(it.second));
            }
        } else {
            LOG(FATAL) << "The input ref is expected to be a Relay function or module";
        }
        return this->code;
    }

  private:
      /*! \brief The function id that represents a C source function. */
     std::string code;
}
```

### Implement a Customized Runtime

接下来，就是实现一个runtime，并将其注册到TVM runtime module里。自定义的runtime应该放在<kbd>src/runtime/contrib/<your-runtime-name>/</kbd>。

首先，自定义的runtime类必须是从<kbd>ModuleNode</kbd>里派生出来的，这样才可兼容。

```c++
#include <dmlc/logging.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <cmath>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {
class ExampleJsonModule : public ModuleNode {
 public:
  explicit ExampleJsonModule(std::string graph_json);

  PackedFunc GetFunction(const std::string& name,
                         const ObjectPtr<Object>& sptr_to_self) final;

  const char* type_key() const { return "examplejson"; }

  void SaveToBinary(dmlc::Stream* stream) final;

  static Module LoadFromBinary(void* strm);

  static Module Create(const std::string& path);

  std::string GetSource(const std::string& format = "");

  void Run(int id, const std::vector<int>& inputs, int output);

  void ParseJson(const std::string& json);

 private:
  /* \brief The json string that represents a computational graph. */
  std::string graph_json_;
  /* \brief The subgraph that being processed. */
  std::string curr_subgraph_;
  /*! \brief A simple graph from subgraph id to node entries. */
  std::map<std::string, std::vector<NodeEntry> > graph_;
  /* \brief A simple pool to contain the tensor for each node in the graph. */
  std::vector<NDArray> data_entry_;
  /* \brief A mapping from node id to op name. */
  std::vector<std::string> op_id_;
};
```

而这个类中需要实现如下函数：

* **Constructor** 接受一个（以ExampleJSON形式表示的）子图，并保存
* **GetFunction** 返回一个封装的函数实现，供TVM runtime执行
* **SaveToBinary**&**LoadFromBinary** 将runtime module化为二进制，以便部署
* **GetSource** 转储生成的代码，给人看的

#### Implement Constructor

```c++
explicit ExampleJsonModule(std::string graph_json) {
  this->graph_json_ = graph_json;
  ParseJson(this->graph_json_);
}
```

其中的<kbd>ParseJson</kbd>，用来解析ExampleJSON格式的子图，并在内存里构造一个图。

```c++
void ParseJson(const std::string& json) {
  std::string line;
  std::string curr_subgraph;
  std::stringstream ss(json);

  while (std::getline(ss, line, '\n')) {
    std::stringstream ss2(line);
    std::string token;
    int id = 0;

    ss2 >> token;
    if (token.find("subgraph_") != std::string::npos) {
      curr_subgraph = token;
      continue;
    }

    ss2 >> id;
    if (op_id_.size() <= static_cast<size_t>(id)) {
      op_id_.resize(id + 1);
      data_entry_.resize(id + 1);
    }

    int64_t total_elements = 1;
    std::vector<int64_t> shape;
    if (token == "input") {
      int64_t size = 0;
      while (ss2 >> size) {
        total_elements *= size;
        shape.push_back(size);
      }
    } else {
      op_id_[id] = token; // 从子图节点ID映射到算子
      bool shape_data = false;
      NodeEntry entry;
      while (ss2 >> token) {
        if (token == "shape:") {
          shape_data = true;
        } else if (shape_data) {
          total_elements *= std::stoll(token);
          shape.push_back(std::stoll(token));
        } else if (token != "inputs:") {
          entry.inputs.push_back(std::stoi(token));
        }
      }
      entry.id = id;
      entry.output = id;
      graph_[curr_subgraph].push_back(entry); //从子图的名称映射到节点数组
    }
    DLContext ctx;
    ctx.device_type = static_cast<DLDeviceType>(1);
    ctx.device_id = 0;
    data_entry_[id] = NDArray::Empty(shape, DLDataType{kDLFloat, 32, 1}, ctx); //从子图节点ID映射到tensor
  }
}
```

#### Implement GetFunction

完成Constructor之后，我们需要实现GetFunction为TVM runtime提供可执行的子图函数。

```c++
PackedFunc GetFunction(const std::string& name,
                       const ObjectPtr<Object>& sptr_to_self) final {
  if (this->graph_.find(name) != this->graph_.end()) {
    this->curr_subgraph_ = name;
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {

      // Copy input tensors to corresponding data entries.
      // 将TVM runtime的数据复制到constructor分配的数据项中
      for (auto i = 0; i < args.size(); ++i) {
        CHECK(args[i].type_code() == kNDArrayContainer || args[i].type_code() == kArrayHandle)
            << "Expect NDArray or DLTensor as inputs\n";
        if (args[i].type_code() == kArrayHandle) {
          DLTensor* arg = args[i];
          this->data_entry_[i].CopyFrom(arg);
        } else {
          NDArray arg = args[i];
          this->data_entry_[i].CopyFrom(arg);
        }
      }

      // Execute the subgraph.
      for (const auto& it : this->graph_[this->curr_subgraph_]) {
        this->Run(it.id, it.inputs, it.output);
      }
      CHECK_GT(graph_.count(this->curr_subgraph_), 0U);

      // Copy the output from a data entry back to TVM runtime argument.
      // 将输出的数据项的结果复制回TVM runtime，输出
      auto out_idx = graph_[this->curr_subgraph_].back().output;
      if (args[args.size() - 1].type_code() == kArrayHandle) {
        DLTensor* arg = args[args.size() - 1];
        this->data_entry_[out_idx].CopyTo(arg);
      } else {
        NDArray arg = args[args.size() - 1];
        this->data_entry_[out_idx].CopyTo(arg);
      }
      *rv = data_entry_.back();
    });
  } else {
    LOG(FATAL) << "Unknown subgraph: " << name << "\n";
    return PackedFunc();
  }
}
```

其中<kbd>Run</kbd>函数的实现如下，主要分为两部分，第一部分是分配一个TVMValue的列表，并映射数据输入；第二部分调用算子。

```c++
void Run(int id, const std::vector<int>& inputs, int output) {
  // Make a list data entry indexs.
  std::vector<int> args(inputs.begin(), inputs.end());
  args.push_back(output);

  // Initialize data holders.
  std::vector<TVMValue> values(args.size());
  std::vector<int> type_codes(args.size());

  // Initialize a TVM arg setter with TVMValue and its type code.
  TVMArgsSetter setter(values.data(), type_codes.data());

  // Set each argument to its corresponding data entry.
  if (op_id_[id] == "add" || op_id_[id] == "sub" || op_id_[id] == "mul") {
    for (size_t i = 0; i < args.size(); i++) {
      setter(i, data_entry_[args[i]]);
    }
  }

  // Invoke the corresponding operator function.
  if (op_id_[id] == "add") {
    Add(values.data(), type_codes.data(), args.size());
  } else if (op_id_[id] == "sub") {
    Sub(values.data(), type_codes.data(), args.size());
  } else if (op_id_[id] == "mul") {
    Mul(values.data(), type_codes.data(), args.size());
  } else {
    LOG(FATAL) << "Unknown op: " << op_id_[id] << "\n";
  }
}
```

最后注册API：

```c++
TVM_REGISTER_GLOBAL("module.examplejson_module_create")
.set_body_typed([](std::string code){
    auto n = make_object<ExampleJsonModule>(code);
    return runtime::Module(n);
});
```

#### Implement SaveToBinary and LoadFromBinary

将构建好的runtime保存到磁盘上时，就需要<kbd>SaveToBinary</kbd>和<kbd>LoadFromBinary</kbd>

首先是<kbd>SaveToBinary</kbd>，在constructor里它接受了**一**个子图表示，这就说明，只需要一个子图就可以构造/恢复这个runtime module，故<kbd>SaveToBinary</kbd>只是把子图写入输出。

```c++
void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(this->graph_json_);
}
```

同理，<kbd>LoadFromBinary</kbd>读取子图的流，在重新构造runtime module

```c++
static Module LoadFromBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string graph_json;
  stream->Read(&graph_json);
  auto n = tvm::runtime::make_object<ExampleJsonModule>(graph_json);
  return Module(n);
}
```

同时我们需要注册<kbd>LoadFromBinary</kbd>来启用Python API。

```c++
TVM_REGISTER_GLOBAL("module.loadbinary_examplejson")
.set_body_typed(ExampleJsonModule::LoadFromBinary);
```



