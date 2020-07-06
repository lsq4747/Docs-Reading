# Relay Pass Infrastructure

Relay优化，每一次pass都是在AST上做Relay-to-Relay的结构转化。文档描述了一种infra设计，可以更系统、更有效地管理pass。

Relay pass infra设计主要受到了以下启发：

* LLVM中的分层pass manager
* 深度学习框架中的模块化的container

Relay pass infra的目标：

* 用户可以灵活地定制自己的optimization pipelines
* 用户可以更好地debug passes
* 减轻了开发人员负担（手动解决各pass之间的依赖关系）
* 简化了新pass的实现

[toc]

* ***前端***  pass infra的主要逻辑

* ***后端***  API供用户交互，可以定制optimization pipelines

  

## C++ Backend

用户注册pass需要用到<kbd>PassInfo</kbd>,其中包含了pass的基本信息：

```c++
class PassInfoNode : public Object {
 public:
  /*! \brief The minimal optimization level that this pass will be enabled. */
  int opt_level;

  /*! \brief The name of an optimization/analysis pass. */
  std::string name;

  /*! \brief The passes that are required to perform the current pass. */
  Array<runtime::String> required;

  PassInfoNode() = default;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("opt_level", &opt_level);
    v->Visit("name", &name);
    v->Visit("required", &required);
  }

  static constexpr const char* _type_key = "transform.PassInfo";
  static constexpr bool _type_has_method_sequal_reduce = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(PassInfoNode, Object);
};
```

*注意：<kbd>required</kbd>是指 要执行当前pass所需要的passes*

```c++
class PassInfo : public ObjectRef {
 public:
  /*!
   * \brief Constructor
   * \param opt_level The optimization level
   * \param name Name of the pass.
   * \param required  The passes that are required to perform the current pass.
   */
  TVM_DLL PassInfo(int opt_level,
                   std::string name,
                   Array<runtime::String> required);

  TVM_DEFINE_OBJECT_REF_METHODS(PassInfo, ObjectRef, PassInfoNode);
};
```



## Pass Constructs

Relay pass infra设计是分层的（LLVM），他引入了一个纯虚的<kbd>PassNode</kbd>:

```c++
class PassNode : public Object {
 public:
  virtual ~PassNode() {}
  /*!
   * \brief Get the pass information/meta data. */
  virtual PassInfo Info() const = 0;

  /*!
   * \brief Transform mod using the default PassContext in the current scope.
   *
   * \param mod The module that an optimization pass runs on.
   *
   * \return The transformed module.
   */
  IRModule operator()(IRModule mod) const {
    return this->operator()(std::move(mod), PassContext::Current());
  }
```

它有以下子类：

* module-level passes
* function-level passes
* <u>sequential passes</u>

### Module-Level Passes

主要针对global and inter-procedural optimizations全局优化与程序间优化，与LLVM中的类似。

```c++
class ModulePassNode : public PassNode {
 public:
  /* \brief The pass meta data.*/
  PassInfo pass_info;

  /*! \brief The pass function sketches the real optimization. For example,
   * we may need to perform dead code elimination on the module level. We could
   * implement the algorithm in the `pass_func` and let it run on a module. It
   * will then remove the dead code including the unused functions in the module.
   */
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func;

  ModulePassNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pass_info", &pass_info);
  }

  /*!
   * \brief Run a module pass on given pass context.
   *
   * \param mod The module that an optimization pass is applied on.
   * \param mod The context that an optimization pass executes on.
   *
   * \return Return the updated module.
   */
  IRModule operator()(IRModule mod, const PassContext& pass_ctx) const final;

  /*!
   * \brief Get the pass information/meta data.
   */
  PassInfo Info() const override { return pass_info; }

  static constexpr const char* _type_key = "transform.ModulePass";
  TVM_DECLARE_FINAL_OBJECT_INFO(ModulePassNode, PassNode);
};
```

其中<kbd>pass_func</kbd>中是具体算法的实现。

### Function-Level Passes

用于给定的模型的函数优化，用于优化relay函数。

```c++
class FunctionPassNode : public PassNode {
 public:
  /* \brief The pass meta data.*/
  PassInfo pass_info;

  /*! \brief The packed pass function sketches the real optimization. For
   * instance, we can implement a pass that works on a Relay function as a
   * `pass_func` and let it run on a given module. The same `pass_func` will
   * then be applied on each function in the module.
   */
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func;

  FunctionPassNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pass_info", &pass_info);
  }

  /*!
   * \brief Run a function pass on given pass context.
   *
   * \param mod The module that an optimization pass is applied on.
   * \param mod The context that an optimization pass executes on.
   *
   * \return Return the updated module.
   */
  IRModule operator()(IRModule mod, const PassContext& pass_ctx) const final;

  /*!
   * \brief Get the pass information/meta data.
   */
  PassInfo Info() const override { return pass_info; }

  static constexpr const char* _type_key = "relay.FunctionPass";
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionPassNode, PassNode);

 private:
  /*
   * \brief Check if a function should be skipped for optimization.
   *
   * \param func The target function to be checked.
   *
   * \return Return true if the function will be skipped, otherwise false.
   */
  bool SkipFunction(const Function& func) const;
};
```

其中，<kbd>SkipFunction</kbd>来控制这个函数在优化时是否被忽略。

### Sequential Passes

它是一些passes的顺序组合，执行时依顺序执行每一个pass。<u>目前Relay只有少数几个passes使用Sequential Pass，如<kbd>FoldScaleAxis</kbd>。</u>

```c++
class SequentialNode : public PassNode {
 public:
  /* \brief The pass meta data.*/
  PassInfo pass_info;

  /*! \brief A list of passes that used to compose a sequential pass. */
  tvm::Array<Pass> passes;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pass_info", &pass_info);
    v->Visit("passes", &passes);
  }

  /*!
   * \brief Get the pass information/meta data.
   */
  PassInfo Info() const override { return pass_info; }

  /*!
   * \brief Check if a pass is enabled.
   *
   * \param info The pass information.
   *
   * \return true if the pass is enabled. Otherwise, false.
   */
  bool PassEnabled(const PassInfo& info) const;

  /*!
   * \brief Resolve the pass dependency. It globs all required passes by
   *        a given pass and executes them.
   *
   * \param mod The module that an optimization pass runs on.
   *
   * \return The updated module after resolving pass dependencies.
   *
   * TODO(zhiics) Build a dependency graph among the passes using provided
   * metadata, i.e. required_passes. Likely, we can have a data structure, i.e.
   * PassInfo, to store the relevant information including the parent passes.
   */
  void ResolveDependency(const IRModule& mod);

  /*!
   * \brief Perform optimizations on a series of passes. The aforementioned
   *        typical pass manager jobs could be done by it. This function could
   *        be overloaded to focus on different metrics, i.e. performance,
   *        memory footprint, etc.
   *
   * \param mod The module that these passes are applied on.
   * \param pass_ctx The context that these passes execute on.
   *
   * \return Return the updated module.
   */
  IRModule operator()(IRModule mod, const PassContext& pass_ctx) const final;

  static constexpr const char* _type_key = "transform.Sequential";
  TVM_DECLARE_FINAL_OBJECT_INFO(SequentialNode, PassNode);
};
```

以下代码显示了在sequential pass中每一个pass如何被实现。先看pass是否被定义，再看是否enable，然后看它的required是否都可用：

```c++
IRModule SequentialNode::operator()(IRModule mod,
                                    const PassContext& pass_ctx) const {
  for (const Pass& pass : passes) {
    CHECK(pass.defined()) << "Found undefined pass for optimization.";
    const PassInfo& pass_info = pass->Info();
    if (!PassEnabled(pass_info))  continue;
    // resolve dependencies
    for (const auto& it : pass_info->required) {
      mod = GetPass(it)(std::move(mod), pass_ctx);
    }
    mod = pass(std::move(mod), pass_ctx);
  }
  return mod;
}
```

其中的<kbd>GetPass</kbd>，检索每一个required pass是否都注册:

```c++
Pass GetPass(const std::string& pass_name) {
  using tvm::runtime::Registry;
  const runtime::PackedFunc* f = nullptr;
  if (pass_name.find("transform.") != std::string::npos) {
    f = Registry::Get(pass_name);
  } else if ((f = Registry::Get("transform." + pass_name))) {
    // pass
  } else if ((f = Registry::Get("relay._transform." + pass_name))) {
  }
  CHECK(f != nullptr) << "Cannot use " << pass_name
                      << "to create the pass";
  return (*f)();
}
```

## Pass Registration

pass需要注册。pass的注册可以分为以下步骤，以<kbd>const folding</kbd>为例:

1. <kbd>Expr</kbd>到<kbd>Expr</kbd>转换

   ```c++
   Expr FoldConstant(const Expr& expr, const IRModule& mod) {
     DLContext ctx;
     ctx.device_type = kDLCPU;
     ctx.device_id = 0;
     Target target = Target::Create("llvm");
     // use a fresh build context
     // in case we are already in a build context.
     With<BuildConfig> fresh_build_ctx(BuildConfig::Create());
   
     return ConstantFolder(CreateInterpreter(mod, ctx, target), mod).Mutate(expr);
   }
   ```

2. 确定是哪个层级的pass（module/function/sequential），如<kbd>const folding</kbd>发生在单个函数上，为<kbd>Function Pass</kbd>：

   ```c++
   namespace transform {
   
   Pass FoldConstant() {
     runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
       [=](Function f, IRModule m, PassContext pc) {
         return Downcast<Function>(FoldConstant(f, m));
     };
     return CreateFunctionPass(pass_func, 2, "FoldConstant", {});
   }
   
   TVM_REGISTER_GLOBAL("relay._transform.FoldConstant")
   .set_body_typed(FoldConstant);
   
   }  // namespace transform
   ```

   同时，pass API端点<kbd>relay._transform.FoldConstan</kbd>被注册。

3. <u>为了适用其他的c++模型，在<kbd>include/tvm/relay/transform.h</kbd>中要声明</u>：

   ```c++
   TVM_DLL Pass FoldConstant();
   ```

## Python Frontend

前端需要一些简单的API来创建和执行pass，后端接收到信息后，决定用哪个函数来创建pass。

1. **PassContext**

   <kbd>PassContext</kbd>包含了pass的信息：

   * 报错系统error reporting system
   * opt level
   * disabled pass
   * required pass
   * fallback device

   <kbd>PassContext</kbd>是为了方便用户在一定的配置下，编写python进行优化来设计的。

   ```python
   @tvm._ffi.register_object("transform.PassContext")
   class PassContext(tvm.runtime.Object):
       """The basis where a Relay optimization/analysis runs on.
       Each pass context contains a number of auxiliary information that is used
       to help an optimization pass. Such information includes the error reporter
       to record the errors of during the optimization, etc.
   
       opt_level : Optional[int]
           The optimization level of this pass.
   
       fallback_device : Optional[Union[int, str, TVMContext]]
           The fallback device type. It is also used as the default device for
           operators that are not annotated during heterogeneous execution.
   
       required_pass : Optional[Union[List[str], Set[str], Tuple[str]]]
           The list of passes that are required by a certain pass.
   
       disabled_pass : Optional[Union[List[str], Set[str], Tuple[str]]]
           The list of passes that are disabled.
       """
       def __init__(self,
                    opt_level=2,
                    fallback_device=_nd.cpu(),
                    required_pass=None,
                    disabled_pass=None,
                    trace=None):
           if isinstance(fallback_device, str):
               fallback_device = _nd.context(fallback_device).device_type
           elif isinstance(fallback_device, tvm.runtime.TVMContext):
               fallback_device = fallback_device.device_type
           if not isinstance(fallback_device, int):
               raise TypeError("fallback_device is expected to be the type of " +
                               "int/str/TVMContext.")
   
           required = list(required_pass) if required_pass else []
           if not isinstance(required, (list, tuple)):
               raise TypeError("required_pass is expected to be the type of " +
                               "list/tuple/set.")
   
           disabled = list(disabled_pass) if disabled_pass else []
           if not isinstance(disabled, (list, tuple)):
               raise TypeError("disabled_pass is expected to be the type of " +
                               "list/tuple/set.")
   
           self.__init_handle_by_constructor__(_ffi_transform_api.PassContext, opt_level,
                                               fallback_device, required,
                                               disabled, trace)
   
       def __enter__(self):
           _ffi_transform_api.EnterPassContext(self)
           return self
   
       def __exit__(self, ptype, value, trace):
           _ffi_transform_api.ExitPassContext(self)
   
       @staticmethod
       def current():
           """Return the current pass context."""
           return _ffi_transform_api.GetCurrentPassContext()
   ```

   c++中的<kbd>PassContext</kbd>:

   ```c++
   class PassContext : public ObjectRef {
    public:
     PassContext() {}
     explicit PassContext(ObjectPtr<Object> n) : ObjectRef(n) {}
     /*!
      * \brief const accessor.
      * \return const access pointer.
      */
     const PassContextNode* operator->() const {
       CHECK(get() != nullptr);
       return static_cast<const PassContextNode*>(get());
     }
     /*!
      * \brief mutable accessor.
      * \return mutable access pointer.
      */
     PassContextNode* operator->() {
       CHECK(get() != nullptr);
       return static_cast<PassContextNode*>(get_mutable());
     }
     /*!
      * \brief Construct a PassContext containing the default configurations.
      * \return The new PassContext.
      */
     TVM_DLL static PassContext Create();
     /*!
      * \brief Get the default pass context in the current scope.
      * \return The pass context.
      */
     TVM_DLL static PassContext Current();
   
     /*!
      * \brief Apply the tracing functions of the context to the module, with the info.
      * \param module The IRModule to trace.
      * \param info The pass information.
      * \param is_before Indicated whether the tracing is before or after a pass.
      */
     TVM_DLL void Trace(const IRModule& module, const PassInfo& info, bool is_before) const;
   
     // accessor.
     using ContainerType = PassContextNode;
     class Internal;
   
    private:
     // The entry of a pass context scope.
     TVM_DLL void EnterWithScope();
     // The exit of a pass context scope.
     TVM_DLL void ExitWithScope();
   
     // Classes to get the Python `with` like syntax.
     friend class Internal;
     friend class With<PassContext>;
   };
   
   ```

2. **Pass**

   对后端实现的pass的封装，例：

   ```python
   def FoldConstant():
       """Fold the constant expressions in a Relay program.
   
       Returns
       -------
       ret : tvm.transform.Pass
           The registered pass for constant folding.
       """
       return _ffi_api.FoldConstant()
   ```

   具体例子见<kbd>tests/python/relay/test_pass_manager.py</kbd>。



自己写了三种pass：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 20:17:30 2020

@author: lsq
"""

import tvm
from tvm import relay
from tvm.relay import transform as _transform


#Create a simple relay program
tp  = relay.TensorType((10,),"float32")
a = relay.var("a",tp) 
b = relay.var("b",tp) 
v = relay.GlobalVar("myAddLog")
mul_add_log = relay.Function([a,b],relay.log(relay.add(a,b)))

x = relay.var("x",tp)
y = relay.var("y",tp)
v_add = relay.GlobalVar("myAdd")
add = relay.Function([x,y],relay.add(x,y))

mod = tvm.IRModule({v: mul_add_log, v_add: add})

#Create a module pass
@_transform.module_pass(opt_level=2)
def transform(expr, ctx):
    tp  = relay.TensorType((10,),"float32")
    x   = relay.var("x",tp)
    gv  = relay.GlobalVar("abs")
    fuc = relay.Function([x],relay.abs(x))
    new = tvm.IRModule({gv: fuc})
    new.update(expr)
    return new

module_pass = transform
assert isinstance(module_pass, _transform.ModulePass)
assert module_pass.info.opt_level == 2


#Create a function pass
@_transform.function_pass(opt_level=1)
class TestReplaceFunc:
    def __init__(self, new_func):
        self.new_func = new_func
    def transform_function(self, func, mod, ctx):
        return self.new_func
    
x = relay.var("x", shape=(10, 20))
f = relay.Function([x], x)
function_pass = TestReplaceFunc(f)
assert isinstance(function_pass, _transform.FunctionPass)
assert function_pass.info.opt_level == 1


#Create a sequential pass
passes = [module_pass, function_pass]
sequential_pass = _transform.Sequential(opt_level=1, passes=passes)

#Debug
def print_ir(mod, info, is_before):
    if is_before:
        print("==========================Pass============================")
        print(mod)
        
with relay.build_config(opt_level=2,trace=print_ir):
    mod = sequential_pass(mod)
    
print("==========================Final=============================")
print(mod)
```

运行结果：

```python
runfile('/home/lsq/TVMLearning/pass.py', wdir='/home/lsq/TVMLearning')
==========================Pass============================
def @myAdd(%x: Tensor[(10), float32], %y: Tensor[(10), float32]) {
  add(%x, %y)
}

def @myAddLog(%a: Tensor[(10), float32], %b: Tensor[(10), float32]) {
  %0 = add(%a, %b);
  log(%0)
}

==========================Pass============================
def @myAddLog(%a: Tensor[(10), float32], %b: Tensor[(10), float32]) -> Tensor[(10), float32] {
  %0 = add(%a, %b) /* ty=Tensor[(10), float32] */;
  log(%0) /* ty=Tensor[(10), float32] */
}

def @myAdd(%x: Tensor[(10), float32], %y: Tensor[(10), float32]) -> Tensor[(10), float32] {
  add(%x, %y) /* ty=Tensor[(10), float32] */
}

def @abs(%x1: Tensor[(10), float32]) {
  abs(%x1)
}

==========================Final=============================
def @abs(%x: Tensor[(10, 20), float32]) -> Tensor[(10, 20), float32] {
  %x
}

def @myAdd(%x1: Tensor[(10, 20), float32]) -> Tensor[(10, 20), float32] {
  %x1
}

def @myAddLog(%x2: Tensor[(10, 20), float32]) -> Tensor[(10, 20), float32] {
  %x2
}
```

