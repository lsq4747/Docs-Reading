# Adding a New Device

1. 补全相应的python接口
2. 找到python和C交互的接口
3. 正确维护中间代码的IR pass变换中新设备引入的特性
4. 代码生成对新设备和新特性的支持
5. 添加编译选项支持(非必须)

[toc]

## 补全相应的python接口

在<kbd>tvm/python/tvm/target.py</kbd>中添加，参考cuda：

```python
def cuda(model='unknown', options=None):
    """Returns a cuda target.

    Parameters
    ----------
    model: str
        The model of cuda device (e.g. 1080ti)
    options : str or list of str
        Additional options
    """
    opts = _merge_opts(['-model=%s' % model], options)
    return _ffi_api.TargetCreate("cuda", *opts)

```

之后在<kbd>runtime_ctypes.py</kbd>的TVMContext中添加：

```python
class TVMContext(ctypes.Structure):
    """TVM context strucure."""
    _fields_ = [("device_type", ctypes.c_int),
                ("device_id", ctypes.c_int)]
    MASK2STR = {
        1 : 'cpu',
        2 : 'gpu',
        4 : 'opencl',
        5 : 'aocl',
        6 : 'sdaccel',
        7 : 'vulkan',
        8 : 'metal',
        9 : 'vpi',
        10: 'rocm',
        11: 'opengl',
        12: 'ext_dev',
        13: 'micro_dev',
        14: 'hexagon',
    }
    STR2MASK = {
        'llvm': 1,
        'stackvm': 1,
        'cpu': 1,
        'c': 1,
        'gpu': 2,
        'cuda': 2,
        'nvptx': 2,
        'cl': 4,
        'opencl': 4,
        'aocl' : 5,
        'aocl_sw_emu' : 5,
        'sdaccel': 6,
        'vulkan': 7,
        'metal': 8,
        'vpi': 9,
        'rocm': 10,
        'opengl': 11,
        'ext_dev': 12,
        'micro_dev': 13,
        'hexagon': 14,
    }
```



## 找到python和C交互的接口

TargetCreate函数通过TVM_REGISTER注册



## 正确维护中间代码的IR pass变换中新设备引入的特性

在target.cc里找到

```c++
Target cuda(const std::vector<std::string>& options) {
  return CreateTarget("cuda", options);
}
```

其中在CreateTarget里面有

```c++
Target CreateTarget(const std::string& target_name,
                    const std::vector<std::string>& options) {
  auto t = make_object<TargetNode>();
  t->target_name = target_name;

  std::string libs_flag = "-libs=";
  std::string device_flag = "-device=";
  std::string keys_flag = "-keys=";
  for (auto& item : options) {
    t->options_array.push_back(item);

    if (item.find(libs_flag) == 0) {
      std::stringstream ss(item.substr(libs_flag.length()));
      std::string lib_item;
      while (std::getline(ss, lib_item, ',')) {
        t->libs_array.push_back(lib_item);
      }
    } else if (item.find(device_flag) == 0) {
      t->device_name = item.substr(device_flag.length());
      t->keys_array.push_back(t->device_name);
    } else if (item.find(keys_flag) == 0) {
      std::stringstream ss(item.substr(keys_flag.length()));
      std::string key_item;
      while (std::getline(ss, key_item, ',')) {
        t->keys_array.push_back(key_item);
      }
    }
  }

  if (t->device_name.length() > 0) {
    t->keys_array.push_back(t->device_name);
  }
  t->device_type = kDLCPU;
  t->thread_warp_size = 1;
  if (target_name == "c" && t->device_name == "micro_dev") {
    t->device_type = kDLMicroDev;
  } else if (target_name == "c" || target_name == "llvm") {
    t->keys_array.push_back("cpu");
  } else if (target_name == "cuda" || target_name == "nvptx") {
    t->device_type = kDLGPU;
    t->keys_array.push_back("cuda");
    t->keys_array.push_back("gpu");
    t->max_num_threads = 1024;
    t->thread_warp_size = 32;
  } else if (target_name == "rocm" || target_name == "opencl") {
    // For now assume rocm schedule for opencl
    if (target_name == "opencl") {
      t->device_type = kDLOpenCL;
    } else {
      t->device_type = kDLROCM;
    }
    t->keys_array.push_back(target_name);
    t->keys_array.push_back("gpu");
    t->max_num_threads = 256;
    if (t->device_name == "intel_graphics") {
      t->thread_warp_size = 16;
    }
  } else if (target_name == "metal" || target_name == "vulkan") {
    if (target_name == "metal") {
      t->device_type = kDLMetal;
    } else {
      t->device_type = kDLVulkan;
    }
    t->keys_array.push_back(target_name);
    t->keys_array.push_back("gpu");
    t->max_num_threads = 256;
  } else if (target_name == "sdaccel") {
    t->device_type = kDLOpenCL;
    t->keys_array.push_back("sdaccel");
    t->keys_array.push_back("hls");
  } else if (target_name == "aocl" || target_name == "aocl_sw_emu") {
    t->device_type = kDLAOCL;
    t->keys_array.push_back("aocl");
    t->keys_array.push_back("hls");
  } else if (target_name == "opengl") {
    t->device_type = kOpenGL;
    t->keys_array.push_back("opengl");
  } else if (target_name == "stackvm") {
    t->device_type = kDLCPU;
  } else if (target_name == "ext_dev") {
    t->device_type = kDLExtDev;
  } else if (target_name == "hybrid") {
    t->device_type = kDLCPU;
  } else if (target_name == "hexagon") {
    t->keys_array.push_back("hexagon");
    t->device_type = kDLHexagon;
  } else {
    LOG(ERROR) << "Unknown target name " << target_name << "; falling back to stackvm";
    return target::stackvm();
  }

  return Target(t);
}
```

> 版本不同
>
> 其中
>
> ```c++
> Target Target::CreateTarget(const std::string& name, const std::vector<std::string>& options) {
> TargetKind kind = TargetKind::Get(name);
> ObjectPtr<TargetNode> target = make_object<TargetNode>();
> target->kind = kind;
> // tag is always empty
> target->tag = "";
> // parse attrs
> target->attrs = target->ParseAttrsFromRaw(options);
> String device_name = target->GetAttr<String>("device", "").value();
> // set up keys
> {
>  std::vector<String> keys;
>  // user provided keys
>  if (Optional<Array<String>> user_keys = target->GetAttr<Array<String>>("keys")) {
>    keys = std::vector<String>(user_keys.value().begin(), user_keys.value().end());
>    target->attrs.erase("keys");
>  }
>  // add `device_name`
>  if (!device_name.empty()) {
>    keys.push_back(device_name);
>  }
>  // add default keys
>  for (const auto& key : target->kind->default_keys) {
>    keys.push_back(key);
>  }
>  // de-duplicate keys
>  target->keys = DeduplicateKeys(keys);
> }
> return Target(target);
> }
> 
> ```
>
> 在<kbd>target_kind.cc</kbd>里
>
> ```c++
> namespace tvm {
> 
> TVM_REGISTER_NODE_TYPE(TargetKindNode);
> 
> TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
>     .set_dispatch<TargetKindNode>([](const ObjectRef& node, ReprPrinter* p) {
>       auto* op = static_cast<const TargetKindNode*>(node.get());
>       p->stream << op->name;
>     });
> 
> using TargetKindRegistry = AttrRegistry<TargetKindRegEntry, TargetKind>;
> 
> TargetKindRegEntry& TargetKindRegEntry::RegisterOrGet(const String& target_kind_name) {
>   return TargetKindRegistry::Global()->RegisterOrGet(target_kind_name);
> }
> 
> void TargetKindRegEntry::UpdateAttr(const String& key, TVMRetValue value, int plevel) {
>   TargetKindRegistry::Global()->UpdateAttr(key, kind_, value, plevel);
> }
> 
> const AttrRegistryMapContainerMap<TargetKind>& TargetKind::GetAttrMapContainer(
>     const String& attr_name) {
>   return TargetKindRegistry::Global()->GetAttrMap(attr_name);
> }
> 
> const TargetKind& TargetKind::Get(const String& target_kind_name) {
>   const TargetKindRegEntry* reg = TargetKindRegistry::Global()->Get(target_kind_name);
>   CHECK(reg != nullptr) << "ValueError: TargetKind \"" << target_kind_name
>                         << "\" is not registered";
>   return reg->kind_;
> }
> 
> ```
>
> ```c++
> TVM_REGISTER_TARGET_KIND("cuda")
>     .add_attr_option<Array<String>>("keys")
>     .add_attr_option<Array<String>>("libs")
>     .add_attr_option<String>("device")
>     .add_attr_option<String>("model")
>     .add_attr_option<Bool>("system-lib")
>     .add_attr_option<Integer>("max_num_threads", Integer(1024))
>     .add_attr_option<Integer>("thread_warp_size", Integer(32))
>     .add_attr_option<String>("mcpu")
>     .set_default_keys({"cuda", "gpu"})
>     .set_device_type(kDLGPU);
> 
> TVM_REGISTER_TARGET_KIND("c")
>     .add_attr_option<Array<String>>("keys")
>     .add_attr_option<Array<String>>("libs")
>     .add_attr_option<String>("device")
>     .add_attr_option<String>("model")
>     .add_attr_option<Bool>("system-lib")
>     .add_attr_option<String>("runtime")
>     .set_default_keys({"cpu"})
>     .set_device_type(kDLCPU);
> ```
>
> 

其中的device_type定义在<kbd>dlpack.h</kbd>里

```c++
typedef enum {
  /*! \brief CPU device */
  kDLCPU = 1,
  /*! \brief CUDA GPU device */
  kDLGPU = 2,
  /*!
   * \brief Pinned CUDA GPU device by cudaMallocHost
   * \note kDLCPUPinned = kDLCPU | kDLGPU
   */
  kDLCPUPinned = 3,
  /*! \brief OpenCL devices. */
  kDLOpenCL = 4,
  /*! \brief Vulkan buffer for next generation graphics. */
  kDLVulkan = 7,
  /*! \brief Metal for Apple GPU. */
  kDLMetal = 8,
  /*! \brief Verilog simulator buffer */
  kDLVPI = 9,
  /*! \brief ROCm GPUs for AMD GPUs */
  kDLROCM = 10,
  /*!
   * \brief Reserved extension device type,
   * used for quickly test extension device
   * The semantics can differ depending on the implementation.
   */
  kDLExtDev = 12,
} DLDeviceType;
```

修改<kbd>device_api.h</kbd>里的case

```c++
inline const char* DeviceName(int type) {
  switch (type) {
    case kDLCPU: return "cpu";
    case kDLGPU: return "gpu";
    case kDLCPUPinned: return "cpu_pinned";
    case kDLOpenCL: return "opencl";
    case kDLSDAccel: return "sdaccel";
    case kDLAOCL: return "aocl";
    case kDLVulkan: return "vulkan";
    case kDLMetal: return "metal";
    case kDLVPI: return "vpi";
    case kDLROCM: return "rocm";
    case kOpenGL: return "opengl";
    case kDLExtDev: return "ext_dev";
    case kDLMicroDev: return "micro_dev";
    case kDLHexagon: return "hexagon";
    default: LOG(FATAL) << "unknown type =" << type; return "Unknown";
  }
}
```

同时，要在<kbd>runtime/module.cc</kbd>的RuntimeEnabled里修改

```c++
bool RuntimeEnabled(const std::string& target) {
  std::string f_name;
  if (target == "cpu") {
    return true;
  } else if (target == "cuda" || target == "gpu") {
    f_name = "device_api.gpu";
  } else if (target == "cl" || target == "opencl" || target == "sdaccel") {
    f_name = "device_api.opencl";
  } else if (target == "gl" || target == "opengl") {
    f_name = "device_api.opengl";
  } else if (target == "mtl" || target == "metal") {
    f_name = "device_api.metal";
  } else if (target == "vulkan") {
    f_name = "device_api.vulkan";
  } else if (target == "stackvm") {
    f_name = "target.build.stackvm";
  } else if (target == "rpc") {
    f_name = "device_api.rpc";
  } else if (target == "micro_dev") {
    f_name = "device_api.micro_dev";
  } else if (target.length() >= 5 && target.substr(0, 5) == "nvptx") {
    f_name = "device_api.gpu";
  } else if (target.length() >= 4 && target.substr(0, 4) == "rocm") {
    f_name = "device_api.rocm";
  } else if (target.length() >= 4 && target.substr(0, 4) == "llvm") {
    const PackedFunc* pf = runtime::Registry::Get("codegen.llvm_target_enabled");
    if (pf == nullptr) return false;
    return (*pf)(target);
  } else {
    LOG(FATAL) << "Unknown optional runtime " << target;
  }
  return runtime::Registry::Get(f_name) != nullptr;
}

```

添加完之后，新建一个目录，加入代码。

## 代码生成对新设备和新特性的支持

在codegen部分加上对这个设备的支持，这样可以在设备上跑c代码。

加CodeGenXXX的class，见Bring U Own Codegen to TVM

## 添加编译选项支持

在cmake/config.cmake里添加新设备的选项，如LLVM、CUDA这些。拷贝到build下。

同时在cmake/module里添加.cmake

在CMakeLists.txt里添加tvm_option和include