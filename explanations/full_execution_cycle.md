```text
                       =======================================
                       FORWARD PASS (MODULE EXECUTION CYCLE)
                       =======================================

  [ 1. Forward Pre-Hook Fires ]
  │
  ├──► Calls `dtensor_param.gather()`
  │      │
  │      └──► Executes `FSDPAllGatherFunction.apply(local_param)`
  │             │
  │             ├──► 1. All-Gather: DTensor[Shard(0)] -> DTensor[Replicate()]
  │             ├──► 2. Inject autograd node: <FSDPAllGatherFunctionBackward>
  │             └──► 3. Assign `full_tensor` to module (e.g. `module.weight = full_tensor`)
  │
  [ 2. Module Executes Computation ]
  │
  ├──► `output = module.forward(input)` using `full_tensor`
  │
  [ 3. Forward Post-Hook Fires ]
  │
  └──► Calls `dtensor_param.free()`
         │
         └──► Swaps module attribute back to `local_param` (Frees `full_tensor` host RAM)


                       =======================================
                       BACKWARD PASS (AUTOGRAD ENGINE TRAVERSAL)
                       =======================================

  [ 4. Loss Backward Triggered ]
  │
  ├──► `loss.backward()` walks backward through autograd computational graph
  │
  [ 5. Autograd Hits Node ]
  │
  ├──► Reaches `<FSDPAllGatherFunctionBackward>` attached to `full_tensor`
  │
  [ 6. Automatic Backward Invocation ]
  │
  └──► PyTorch C++ Engine calls `FSDPAllGatherFunction.backward(grad_output)`
         │
         ├──► 1. Incoming `grad_output` wrapped as `DTensor[Partial()]`
         ├──► 2. Reduce-Scatter: `DTensor[Partial()]` -> `DTensor[Shard(0)]`
         ├──► 3. Scale by world size: `sharded_grad = sharded_grad / world_size`
         └──► 4. Return `sharded_grad`
                │
                └──► PyTorch Autograd automatically assigns `sharded_grad`
                     directly to `local_param.grad`

```


```text

========================================================================================
                                    FORWARD PASS
========================================================================================

 [ Forward Pre-Hook ] ──► All-Gather #1 ──► Module Computation ──► [ Forward Post-Hook ]
 (Gathers Full Weight)                       (using Full Weight)   (Frees Full Weight)

========================================================================================
                                    BACKWARD PASS
========================================================================================

 [ Autograd reaches Layer ]
           │
           ▼
 [ Pre-Backward Hook / Re-Gather ]  ◄─── THIS WAS MISSING!
           │
           ├──► Executes All-Gather #2 (Re-creates Full Weight $W$)
           │
           ▼
 [ Module Matrix Math ]
           │
           ├──► Calculates $\nabla W = X^T \cdot \nabla Y$  (Creates `grad_output`)
           │
           ▼
 [ FSDPAllGatherFunction.backward(grad_output) ]
           │
           ├──► 1. Takes `grad_output` (Full Gradient)
           ├──► 2. Runs Reduce-Scatter ──► Produces `sharded_grad`
           ├──► 3. Assigns `sharded_grad` to `local_param.grad`
           └──► 4. Frees Full Weight $W$ again

```
