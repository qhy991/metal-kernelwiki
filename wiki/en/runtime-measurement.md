# Runtime validation and GPU timing with limited tools

[中文](../runtime-measurement.md) · English companion edition

Offline Metal tools, existing framework operators and runtime source compilation are separate capabilities. Preserve the current developer directory, SDK, versions and errors. An offline lookup failure does not prove runtime compilation fails; a built-in operator passing does not prove custom kernels work. [Metal libraries](https://developer.apple.com/documentation/metal/metal-libraries)

Check import, device, library, pipeline and execution independently. File existence is not runnable-tool evidence. Compile-only probes establish source/function/pipeline creation, while execution probes additionally submit work and check outputs. Preserve failed routes rather than repairing the environment and calling the repaired state the original result.

`gpuStartTime` and `gpuEndTime` are seconds, read after completion and error checks. Their difference measures the command buffer's GPU interval, not an individual kernel automatically. `kernelStartTime`/`kernelEndTime` are CPU scheduling timestamps. Reject zero, reversed and nonfinite times. [GPU timestamps](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpuendtime)

Enumerate counter support and sampling boundaries. Stage and dispatch boundaries differ. Resolve samples after completion; check length, zero values and `MTLCounterErrorValue`. Raw GPU timestamps are not necessarily nanoseconds. Use two valid `sampleTimestamps()` reference pairs:

~~~text
elapsed_ns = sample_GPU_span / reference_GPU_span × reference_CPU_span
~~~

Calibration and profiling add overhead; ordinary benchmarks run separately. [Clock conversion](https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time)

On macOS 14+, documented per-process `MTL_CAPTURE_ENABLED=1` supports programmatic capture. Check trace support, start, commands, stop and export. An exported file does not prove replay or analysis succeeds. `parsed=false` means unparsed, not necessarily previously failed or permanently unavailable. [Capture API](https://developer.apple.com/documentation/xcode/capturing-a-metal-workload-programmatically)

Predeclare bounded shapes, memory, warmups, samples and total work. Check an external oracle before timing. Preserve failures and invalid times without automatic larger retries. Capture-only work leaves correctness and ordinary performance unevaluated unless separately performed. Report each capability and coverage boundary explicitly.

## Registered sources

These are shared with the Chinese record. Source metadata retains its original language.

- [apple-metal-libraries](https://developer.apple.com/documentation/metal/metal-libraries) (official-doc)
- [apple-metal-offline-compilation](https://developer.apple.com/documentation/metal/building-a-shader-library-by-precompiling-source-files) (official-doc)
- [apple-commandbuffer-gpu-time](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpuendtime) (official-doc)
- [apple-commandbuffer-debugging](https://developer.apple.com/documentation/metal/command-buffer-debugging) (official-doc)
- [apple-counter-sampling-boundaries](https://developer.apple.com/documentation/metal/sampling-gpu-data-into-counter-sample-buffers) (official-doc)
- [apple-counter-resolution](https://developer.apple.com/documentation/metal/converting-a-gpus-counter-data-into-a-readable-format) (official-doc)
- [apple-counter-clock-conversion](https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time) (official-doc)
- [apple-programmatic-capture](https://developer.apple.com/documentation/xcode/capturing-a-metal-workload-programmatically) (official-doc)

[Shared source catalog](../../data/catalog.json)
