# Metal LLM profiling and evidence

[中文](../metal-profiling.md) · English companion edition

Start with the request timeline, then inspect the critical pipeline. These collection plans have no new local execution claim.

Metal System Trace locates CPU encoding, GPU work, memory activity and waits. Xcode Metal debugging inspects captured resources, pipelines, shaders and counters. Current [Metal tools](https://developer.apple.com/metal/tools/) include gpucapture, gpudebug and metalperftrace; check installed help and Xcode before promising availability.

~~~text
request trace → critical interval → actual pipeline → limiter/counter
→ falsifiable hypothesis → correctness → timing without capture
~~~

Capture representative prefill and decode separately. Record lengths, batch/context, weight/KV dtype, cache state and timing boundary. Separate load/JIT/warmup from steady inference. CPU enqueue duration is not GPU completion time.

| Observation | Investigate |
|---|---|
| GPU idle, CPU busy | Encoding, submission, synchronization and object reuse |
| Low occupancy, saturated ALU | ALU work before occupancy tuning |
| Low occupancy and ALU | Grid/resources, cache and MMU with supported counters |
| Fusion regression | Live registers, stack spills, cache pressure, smaller tiles |

Large command batches can reduce overhead while increasing queueing latency. Metal 4's queue model also changes synchronization responsibility: resource conflicts require correct barrier/fence/event handling. [Metal 4 API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api)

Enumerate supported counter sets and sampling locations before collecting. Clock alignment may require conversion. [Counter API](https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers)

Profiling explains a measurement; it does not replace controlled, capture-free latency samples. Retain configuration, oracle/tolerance, candidate hypothesis, raw samples, trace paths, counter gaps and end-to-end results. Fix power/concurrency/cache conditions, distinguish independent throughput from dependent token latency, and preserve environment/tool failures separately from later successful routes.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [apple-metal-tools](https://developer.apple.com/metal/tools/) (official-doc)
- [apple-command-buffers](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/CommandBuffers.html) (official-doc)
- [apple-m3-profiling](https://developer.apple.com/videos/play/tech-talks/111374/) (official-doc)
- [apple-m5-profiling](https://developer.apple.com/videos/play/tech-talks/111431/) (official-doc)
- [apple-metal4-core](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api) (official-doc)
- [apple-resource-sync](https://developer.apple.com/documentation/metal/resource-synchronization) (official-doc)
- [apple-gpu-counters](https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers) (official-doc)

[Shared source catalog](../../data/catalog.json)
