# Release scope

OpenFlyScan is the main entry point for the workstation service, Quality Predictor
and training/evaluation code. Android V4, Android V5, iOS and UE are separate
repositories under `mistletoe235`; their links are in [the README](../README.md).
All five source repositories are public. HF package access is awaiting resolution
of the organization's public-storage quota.

## Publicly available

- **Main repository:** trained head, small real-feature inference example,
  training/evaluation code, workstation API, Pi3X feature pipeline and schema
  13/14 reacquisition planning/export.
- **Android releases:** signed V4 and V5 APKs, checksums and third-party notices
  in both the main and matching app releases. See [installation](apps.md).
- **iOS:** source and a [TestFlight request discussion](https://github.com/mistletoe235/OpenFlyScan/discussions/1),
  not an installable IPA or a guaranteed available TestFlight build.

The Expo East runtime and its license/source attachments are uploaded to HF,
but public download is not available yet; see [download status](simulator.md).

The head ships at `weights/quality_predictor.pt`; it does not require an HF
download. The HF package contains converted scene resources, not standalone GS
PLYs. Full training images, teacher grids and DINO caches are not bundled, and
the downloadable training bundle has not yet been published. Training with your
own prepared data is described in [the training guide](training_data.md).

## Source and release records

Checkpoint compatibility and original import mappings are recorded in
`source_import.json` and [naming](naming.md). Published package manifests identify
their source revisions; later documentation commits do not change those binaries.
See [validation](validation.md) for completed checks and
[current release status](release_readiness.md) for outstanding work.

Original code, the head and the inference example are Apache-2.0; third-party
code, backbone weights and simulator assets retain their respective terms.
See [licensing](licensing.md).
