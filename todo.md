### High level UML
         hardware config
                |
                v
Interface -> planner -> executor
### V1 Repo Struct
do not split include & src... dump both headers and cpp into src and use cmake include
- interface
- planner: planners, plan, cache
- cpu: 
  + cpu's interface
  + radix
  + schedule
  + parallelism (can be enable / disable via a flag)

- gpu: 
  + naive-only for now


# TODO
- Test scripts + benchmark scripts
- Generalized N
- Four-step N1 x N2 planning
- Tiling planning later
- In-place support: FFT(vec) instead of out = FFT(in)
- Optimize CUDA
- Technically if we do radix-2 and radix-3 then we can do everything.