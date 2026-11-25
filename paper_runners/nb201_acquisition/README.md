# Preserved main NAS-Bench-201 acquisition builders

These byte-preserved sources expose the acquisition chain used to prepare the
main replication. They are **not** called by `run_all.py`, and they are not a
turnkey training replay. Some actions prepare or submit Kubernetes Jobs. Do not
execute them merely to inspect the protocol, and do not point them at existing
outputs without checking the original contracts and preservation guards.

| Source | SHA-256 |
|---|---|
| `nas201_taylor_pilot_v1.py` | `87b2357f87038bd6206243962dbef149a0c55557ffcf2f05b1996c0badf54e5f` |
| `nb201_v1/launch.py` | `85b62fd53d25c55a18a094e8eb4463419fc7d2ddf1eb58fac459c8f0c4d327e9` |
| `nb201_v1/full.py` | `8565d3fcc96054eaab8bd3f8e4a811a50b01b3230eb14de2e9e92eb23cca9cfc` |
| `nb201_v1/cifar100.py` | `809d929e74f414a704cf751894c0b999d0a331bdb95535b5fb659f751b0b0095` |
| `nb201_v1/imagenet16.py` | `8fd5ad6c1b23e64f873f22158bfd5d34da1cbee99a084eddb0ea5bedcc141d2b` |

## Source chain and settings

- The shared `measure_taylor` deep-copies the candidate, creates mean-reduced
  `CrossEntropyLoss()`, and calls `taylor_importance([batch], ..., num_batches=1)`.
- CIFAR-10-valid's `launch.py` constructs a fixed augmented 256-image training
  batch. `full.py` verifies/imports that worker for full collection.
- CIFAR-100's builder verifies the shared runner, then derives `engine.py` by
  changing its collection-count check. ImageNet16's builder applies its explicit
  dataset/model adaptations and records Taylor batch size 256.
- All paths pin the historical extractor SHA-256
  `f19dcb88702679de1ada97dde6766bbbdb44019d4b05c04e1557fa1bdde9ad9f`.
  Exact bytes are available at the immutable upstream/fork commit:
  https://github.com/trisrota-commits/nn-dataset/blob/d2c75d7122c6a8b14afbfb43ac392c7e294ba7ac/ab/nn/util/LayerAnalysis.py
  Its Taylor method explicitly switches the copied network into training mode.

## What this addition does not recover

The original paths `nb201-v1-cifar10-pilot/runner.py`,
`nb201-v1-cifar100-full/engine.py` and `nb201-v1-imagenet16-full/engine.py`
are prepared runtime products/dependencies. These builders show their provenance
and transformations; this addition is **not** a claim that every original
prepared engine, full-run contract, dataset, candidate checkpoint, or runtime
receipt has been recovered. Candidate data, benchmark API/snapshot, official
xautodl source/configs, original preparation artifacts, GPU environment, mounts
and cluster permissions are external dependencies retained in the source checks.

The existing saved-prediction reproduction remains CPU-only and independent of
these historical launchers. Original training/acquisition has not been rerun.
Named historical filesystem provenance is preserved; this is not an anonymous
review bundle, and no blanket relicensing of upstream code/data is asserted.
