# Licensing

## OpenFlyScan code and released head

Original OpenFlyScan code, documentation, the released Quality Predictor head
and the small inference example are licensed under Apache-2.0. The full text is
in [LICENSE](../LICENSE). Third-party files retain their original terms and
notices; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

This source license is not a blanket license for every dependency, SDK, model,
scene or packaged simulator used with the project.

## Backbone weights

Pi3's official source code uses BSD-3-Clause, whereas the official Pi3/Pi3X
weights use CC BY-NC 4.0. UAVFF3D's project-page license does not independently
license its pretrained or fine-tuned checkpoints for unrestricted commercial use.
Use the terms attached to the exact checkpoint or obtain separate permission.
A model download link does not grant a different license.

The OpenFlyScan head is distributed separately and contains no backbone weights.
Its Apache-2.0 license does not remove restrictions on external weights used to
extract its inputs.

## Apps, simulator and data

The App and UE repositories license their own code separately. DJI SDK binaries
remain under DJI's EULA; Unreal Engine remains under Epic's EULA. Original
project code being open source does not relicense either SDK or engine.

Large GS scenes and simulator packages stay outside this repository. Their
asset notices and applicable upstream licenses must travel with those packages.
The project does not assign one blanket Apache license to the HF dataset.

## Upstream references

- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt)
- [Pi3 code and model license table](https://github.com/yyfz/Pi3#-license)
- [Pi3X model card](https://huggingface.co/yyfz233/Pi3X/blob/main/README.md)
- [UAVFF3D license scope](https://github.com/yanxian-ll/UAVFF3D#license)
- [DJI SDK EULA](https://developer.dji.com/policies/eula/)
- [Unreal Engine EULA](https://www.unrealengine.com/en-US/eula/unreal)
