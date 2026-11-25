# Historical Taylor acquisition settings

This is an implementation-provenance note, not a new GPU acquisition receipt.
The saved-prediction reproduction commands do not rerun candidate training.

## LEMUR

The historical training wrapper constructs `train_loader_f(train_dataset, batch, ...)`
and calls `full_analysis(self.train_loader, self.loss_fn, num_batches=1)`.
The Taylor method explicitly switches its model to training mode, measures one
minibatch, and restores the prior mode. Its nominal batch size is therefore the
candidate-specific training batch, not a universal 32 or 128. A short dataset or
custom collator can affect the actual tensor size.

The extractor accepts the caller's loss function. The wrapper prefers a model's
`criterion`, then `loss_fn`; classification falls back to `CrossEntropyLoss()`
(default mean reduction). Consequently the implementation does **not** establish
that every heterogeneous stored record used the same loss or reduction.
The exact extractor revision/effective runtime conditions of every historical
LEMUR record are not fully attested by the predictor snapshot.

Immutable historical sources:
- [Train.py](https://github.com/trisrota-commits/nn-dataset/blob/d2c75d7122c6a8b14afbfb43ac392c7e294ba7ac/ab/nn/util/Train.py#L236-L249)
- [Loader construction](https://github.com/trisrota-commits/nn-dataset/blob/d2c75d7122c6a8b14afbfb43ac392c7e294ba7ac/ab/nn/util/Util.py#L178-L181)
- [Taylor extraction](https://github.com/trisrota-commits/nn-dataset/blob/d2c75d7122c6a8b14afbfb43ac392c7e294ba7ac/ab/nn/util/LayerAnalysis.py#L209-L259)

## NAS-Bench-201 main replication

The preserved main-replication acquisition chain uses one fixed augmented
256-image training batch with real labels per dataset, reused at early checkpoints,
and `CrossEntropyLoss()` (mean reduction). The caller deep-copies the candidate;
the pinned Taylor method explicitly calls `.train()` on that copy. Deep-copying
protects the original state but does not imply evaluation-mode measurement.
This applies to the main CIFAR-10-valid, CIFAR-100 and ImageNet16-120 acquisition
implementation, not the later controlled loss/confidence study or superseded
`original/collect.py`.

The pinned extractor bytes match the immutable historical source linked above:
`f19dcb88702679de1ada97dde6766bbbdb44019d4b05c04e1557fa1bdde9ad9f`.
The historical shared caller hash is
`87b2357f87038bd6206243962dbef149a0c55557ffcf2f05b1996c0badf54e5f`.

Source/contract verification is stronger than assuming today's defaults, but it
is not an independently reconstructed per-candidate runtime attestation for every
original training job. The current repository-root `LayerAnalysis.py` is a later
supplied definition file, not the hash-pinned historical NB201 extractor.

## Controlled acquisition

The 32/128 comparison is a separate factorial measurement study. Its results do
not retrospectively establish a universal extraction size, reduction, or mode
for all stored LEMUR values. Measured extraction overhead is not end-to-end NAS
savings. See the preserved `paper_runners/point4/` sources and evidence reporter.
