# Source-release and licensing review — September 21, 2026

[English](license_review_20260921.md) · [Chinese reference](license_review_20260921.zh-CN.md)

This dated record describes the preparation state and evidence from that review.
The five repositories have since been pushed privately under `mistletoe235`; see
[repository links](../components.json). Private publication does not resolve the
remaining source-provenance or asset-licensing questions.

## Conclusion

Original main-repository code, the released head and small sample explicitly use
Apache-2.0. Original App/UE work retains its existing Apache-2.0 terms. Third-party
components, backbone weights and scene assets retain separate terms. This is not
a declaration that the full rights chain of every repository and asset is settled;
the main remaining question is the initial UE source provenance. Paper-by-paper
GS inventory reconciliation was not added. The review itself did not create
remotes, commit, push or change visibility.

## State at review time

| Component | Established | Remaining publication issue |
| --- | --- | --- |
| Main | Root LICENSE, head/sample notices and Python license metadata added | GitHub destination, final revision and private security contact |
| Android V4 | Apache-2.0 LICENSE, safety guidance and third-party notices | Project licensing does not replace SDK/dependency or package terms |
| Android V5 | Apache-2.0 plus retained DJI SDK/sample terms and FFmpeg LGPL-2.1 | Matching notices/source-availability information for APK distribution |
| iOS | Apache-2.0; dependencies supplied through CocoaPods/Swift Packages | SDK terms and signed-package distribution handled separately |
| UE | Original code declares Apache-2.0; full MPL text and exact Eigen source included by new packaging; release-tree check passed | NanoGS headers, exact package/source correspondence and engine-distribution scope |

At review time, all five repositories lacked Git remotes and contained uncommitted
changes. The main/App security guides lacked a private reporting contact; UE gained
`SECURITY.md` and HIL safety guidance. These are publication/maintenance issues,
not algorithm defects.

Bounded scans found no matching private keys, HF/GitHub tokens, signing files or
files over the project's 50 MiB threshold: main 117 files; V4 277 files/260 history
blobs; V5 2,156/2,105; iOS 152/131. This was not a complete credential or legal audit.

## Upstream license boundaries

### Pi3X code and weights have different licenses

- Official Pi3 code uses BSD-3-Clause.
- The official README lists Pi3/Pi3X weights under CC BY-NC 4.0, and the Pi3X model card declares `license: cc-by-nc-4.0`.
- UAVFF3D's Apache-2.0 statement covers its documentation/project pages; its README separately notes that data, 3D assets, checkpoints and third-party model code may use other licenses.
- An Apache-2.0 file in GeoFF3D/UAVFF3D does not establish unrestricted commercial rights to the specific UAVFF3D-finetuned Pi3X checkpoint.

Original-code licensing does not replace backbone-weight licensing. Do not promise
commercial use of the complete pipeline with noncommercial weights without separate
authorization covering the actual checkpoint.

The approximately 5 MB head contains no Pi3X backbone parameters. Use of Pi3X
features alone is not sufficient to conclude that it automatically inherits
CC BY-NC; the head's separate authorization also depends on original code, training
assets and applicable agreements. This pass declared Apache-2.0 for the head in
`../weights/README.md` and the asset manifest, without relicensing external weights
or input data.

### DJI sample code and SDK binaries are separate

The V5 official license distinguishes MIT sample code from SDK binaries governed
by the DJI EULA, and separately notes the LGPL-2.1 terms for dynamically linked
FFmpeg. Apache-2.0 for original App code does not relicense SDK/package dependencies.
Installation packages need applicable third-party notices and source-availability
instructions.

### UE project code is not engine code

The NanoGS/project source and shaders contained 102 files with Epic copyright
headers, including `Plugins/NanoGS/Shaders/Private/GaussianSplatting.ush` and
`Plugins/NanoGS/Source/NanoGS/Public/Paged/NanoGSTreeSelector.h`. These may reflect
template history or actual source provenance; headers alone do not establish which.
Do not remove them in bulk or declare every file original without checking sources.

Epic's EULA distinguishes packaged products, Engine Code and Engine Tools. Some
Editor/Developer code and development tools have distribution-channel restrictions.
A custom Editor module alone does not establish a violation. Distinguish original
plugin code, engine components actually shipped, and the runtime PLY converter
when reviewing the specific package.

The packaging scripts already copied LICENSE, THIRD_PARTY_NOTICES.md and LICENSES/.
Eigen retained MPL references; the full MPL text is now present, and new packages
include corresponding source at `THIRD_PARTY_SOURCES/eigen3.tar.gz` with instructions.
These requirements apply to MPL-covered components, not automatically the entire
UE project. The existing Expo East package records a base revision plus uncommitted
changes; future releases should correspond to fixed commits.

## Licensing arrangement

1. Original main code: Apache-2.0, matching original App/UE work.
2. Original head: explicit Apache-2.0 after confirming release rights, with a statement that backbone weights are excluded.
3. Third-party code, SDKs and models: retain their licenses/copyrights instead of applying the root license to them.
4. GS/simulator packages: retain asset and engine terms; do not label the entire HF dataset with the source-code license.
5. A person authorized to represent the authors/institution should confirm release authority for original work.

## Primary sources consulted in the original review

- [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0.txt), especially sections 3, 4, 7 and 8.
- [Pi3 code license](https://github.com/yyfz/Pi3/blob/main/LICENSE).
- [Pi3 README license table](https://github.com/yyfz/Pi3#-license).
- [Pi3X model card](https://huggingface.co/yyfz233/Pi3X/blob/main/README.md).
- [UAVFF3D README](https://github.com/yanxian-ll/UAVFF3D#license).
- [DJI V5 LICENSE.txt](https://github.com/dji-sdk/Mobile-SDK-Android-V5/blob/dev-sdk-main/LICENSE.txt).
- [DJI EULA](https://developer.dji.com/policies/eula/).
- [CC BY-NC 4.0 legal text](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en).
- [Unreal Engine EULA](https://www.unrealengine.com/en-US/eula/unreal), sections 4–7; direct access returned 403, so the original review read the official page through a text proxy.
- [MPL-2.0](https://www.mozilla.org/en-US/MPL/2.0/), sections 3.1–3.3.

This is a technical release/licensing review, not a legal guarantee of the full
rights chain. Maintainers or the institution must resolve uncertain provenance
and authorization.

## Implementation and recorded checks

- Main: Apache-2.0 LICENSE, retained Pi3/COLMAP licenses, head/sample notices and wheel-packaged license texts. Backbone noncommercial terms are separate.
- Apps: full DJI SDK/sample and FFmpeg LGPL-2.1 terms retained. iOS also retains Clipper2 and installed DJIWidget terms; a stale ZIPFoundation dependency description was removed.
- UE: full MPL, security/provenance records and exact Eigen source packaging added. A temporary package's 337 Eigen files matched repository source byte-for-byte. The existing HF package was not rebuilt or replaced.
- UE header history: 50 paths appeared in the legacy initial import, 42 were added later and 10 lacked a matching historical path. Per-file records are in UE `Docs/NANOGS_SOURCE_PROVENANCE_20260921.json`; no copyright headers were removed.
- The Python wheel built with the root license, third-party notices and three retained license texts. Main/App bounded current/history scans and the UE release-tree check passed.

NanoGS initial-import/renamed-file provenance and a usable private security contact
remain to be confirmed. GitHub destinations have since been set. Public-source
release and formal binary/scene-package distribution remain separate steps.
