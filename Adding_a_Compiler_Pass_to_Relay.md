# Adding a Compiler Pass to Relay

一个Compiler pass可以修改AST、收集AST信息。

编写一个pass有两个关键步骤：

* **创建**c++类，可以遍历程序
* 将遍历的实现和它的metadata**封装**到pass manager API里，让它可以和 relay pass infra对接

[toc]

## AST Traversers

遍历程序的基类是<kbd>ExprFunctor</kbd>,它的接口是<kbd>VisitExper</kbd>。

每一个<kbd>VisitExper_</kbd>定义都针对一个特定类型的表达式，为了得知访问的节点的类型，<kbd>ExprFunctor</kbd>提供了<kbd>VisitExper</kbd>：

```c++
 virtual R VisitExpr(const PrimExpr& n, Args... args) {
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  virtual R VisitExpr_(const VarNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const CallNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const AddNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const SubNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const MulNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const DivNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
```

<kbd>ExprFunctor</kbd>是一个非常通用的类，他有子类<kbd>ExprVisitor</kbd>和<kbd>ExprMutator</kbd>。

### Expression Visitors

<kbd>ExprVisitor</kbd>用于程序分析和收集信息的pass，不是修改程序的pass。<kbd>ExprVisitor</kbd>的<kbd>VisitExpr_</kbd>只是访问了表达式的表达字段。

```c++
void ExprVisitor::VisitExpr_(const LetNode* op) {
  this->VisitExpr(op->value);
  this->VisitExpr(op->var);
  this->VisitExpr(op->body);
}

void ExprVisitor::VisitExpr_(const IfNode* op) {
  this->VisitExpr(op->cond);
  this->VisitExpr(op->true_branch);
  this->VisitExpr(op->false_branch);
}
```

### Expression Mutators

<kbd>ExprMutator</kbd>用于转换程序的pass。

```c++
Expr ExprMutator::VisitExpr_(const VarNode* op) {
  if (op->type_annotation.defined()) {
    auto type = this->VisitType(op->type_annotation);
    if (!op->type_annotation.same_as(type)) {
      return Var(op->vid, type);
    }
  }
  // default case return self.
  return GetRef<Expr>(op);
}
```

### Example：Constant Folding

Constant Folding包含一个visitor<kbd>ConstantChecker</kbd>和一个mutator<kbd>ConstantFolder</kbd>。

#### ConstantChecker

用于检查表达式是否为常量（ConstantNode或者是只有常量的TupleNode）。ConstantNode是直接check，Tuple是将它缓存在<kbd>memo_</kbd>中，再check：

```c++
class ConstantChecker : private ExprVisitor {
 public:
  // Check whether an expression is constant. The results are memoized.
  bool Check(const Expr& expr) {
    // The `ConstantNode` case is common enough that we check directly for the
    // case here, to avoid the time overhead of dispatching through the vtable
    // and the space overhead of memoizing always-true results.
    if (expr.as<ConstantNode>()) {
      return true;
    }
    const auto it = memo_.find(expr);
    if (it != memo_.end())
      return it->second;
    VisitExpr(expr);
    return memo_[expr];  // return memoized result or the default value false
  }

 private:
  std::unordered_map<Expr, bool, ObjectHash, ObjectEqual> memo_;

  void VisitExpr_(const TupleNode* n) final {
    bool result = true;
    for (const auto& field : n->fields) {
      if (!Check(field)) {
        result = false;
        break;
      }
    }
    memo_[GetRef<Tuple>(n)] = result;
  }
};
```

#### ConstantFolder

LetNode、TupleItemGetNode、CallNode这三种节点类型都涉及到常量折叠：

在**LetNode**情况下，我们先尝试折叠value，如果可以，就将它返回到body上。

```c++
  Expr VisitExpr_(const LetNode* op) final {
    Expr value = this->Mutate(op->value);
    if (value.as<ConstantNode>()) {
      memo_[op->var] = value;
      return this->Mutate(op->body);
    } else {
      Var var = Downcast<Var>(this->Mutate(op->var));
      Expr body = this->Mutate(op->body);
      if (var.same_as(op->var) &&
          value.same_as(op->value) &&
          body.same_as(op->body)) {
        return GetRef<Expr>(op);
      } else {
        return Let(var, value, body);
      }
    }
  }
```

在**TupleItemGetNode**情况下，先检查<kbd>op->tuple</kbd>是不是一个TupleNode，因为有些时候本身不是Tuple的<kbd>op->tuple</kbd>可能被评估为Tuple。

```c++
  Expr VisitExpr_(const TupleGetItemNode* op) final {
    Expr res = ExprMutator::VisitExpr_(op);
    op = res.as<TupleGetItemNode>();
    if (const auto* tuple = op->tuple.as<TupleNode>()) {
      return tuple->fields[op->index];
    } else {
      return res;
    }
  }
```

在**CallNode**情况下，只有在所有参数都是常量的情况下才会使用ConstantChecker：

```c++
  Expr VisitExpr_(const CallNode* call) final {
    static auto op_stateful = Op::GetAttr<TOpIsStateful>("TOpIsStateful");

    std::unordered_set<std::string> skip_list{"zeros_like", "ones_like", "full_like", "full"};

    auto origin_args = call->args;
    Expr res = ExprMutator::VisitExpr_(call);
    call = res.as<CallNode>();
    // We don't constant fold function with zero arguments.
    // This is a heuristic that is useful.
    // For example it is harmful to fold ones(shape=(4, 5)).
    if (call->args.size() == 0) return res;
    const OpNode* op = call->op.as<OpNode>();
    if (op == nullptr) return res;
    if (skip_list.count(op->name)) {
        return res;
    }
    // skip stateful ops.
    if (op_stateful.get(GetRef<Op>(op), false)) return res;
    // Try to evaluate shape_of op
    if (call->op == shape_of_op_) {
      return EvaluateShapeOf(res, origin_args, call->attrs);
    }

    // We should think about potentially constant evaluation over these ops too.
    if (call->op == invoke_tvm_op_ ||
        call->op == shape_func_op_ ||
        call->op == alloc_tensor_op_ ||
        call->op == alloc_storage_op_) {
      return GetRef<Call>(call);
    }
      
    bool all_const_args = true;
    for (Expr arg : call->args) {
      if (!checker_.Check(arg)) {
        all_const_args = false;
      }
    }
    if (all_const_args) {
      return ConstEvaluate(res);
    } else {
      return res;
    }
  }
```

## Registering a Pass with the Pass Manager

编写好AST遍历之后，下面代码可以把pass注册成TVM的API端点：

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

<u>FoldConstant does not have any dependencies, but many Relay passes do depend on having type information, so InferType is a common dependency; others may depend on the program’s being in A-normal form, via the ToANormalForm pass.</u>

一旦用上述方式定义了pass，就可以用于sequential pass构造了。

关于注册的更多信息，在 TVM Runtime System。