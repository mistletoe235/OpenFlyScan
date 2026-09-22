# Five-repository release preparation

The source layout is fixed: one OpenFlyScan main repository, Android V4, Android
V5, iOS and UE. The workstation service belongs in the main repository. The
existing website and HF dataset are separate publication surfaces, not additional
algorithm repositories.

## Prepared in this pass

- Android V5 explicitly negotiates continuous schema 14 when requested.
- Android V4/V5 allow read-only route preview without granting execution approval.
- Opt-in native-client tests exercise a configured workstation without aircraft commands.
- A self-contained CPU example checks the main predictor interface.
- Original source, the released head and bundled example have explicit Apache-2.0 licensing; third-party and backbone terms remain separate.
- GeoFF3D base revision, runtime patch and selected environment versions are recorded.
- A portable training-plan template/resolver retains the frozen data checks.
- UE uses the OpenFlyScan public name while retaining compatible project filenames.
- Source checks report large files, signing material and recognizable credential tokens
  without printing secret values.

## Remaining publication decisions

| Item | Required action |
| --- | --- |
| Security contact | Supply a private reporting address for all five repositories |
| GitHub destinations | Five private repositories under `mistletoe235`; source links are in `components.json` |
| Release binaries | Tag the intended source revision before building distributable packages; private source publication does not publish binaries |
| SDK activation/signing | Supply private DJI/map keys and Android/iOS signing credentials for distributable apps |
| Model and scene assets | Publish the head/sample with source; upload GS and simulator assets separately to HF |
| Training reproduction | Release the teacher/cache bundle and verify direct training and checkpoint resume |
| UE source provenance | Resolve retained Epic copyright headers and confirm the specific source/package distribution scope |
| UE binary correspondence | Build/tag from the finalized source and preserve exact archive checksums and manifests |

The predictor checkpoint and small real-feature example are included in the local
source tree. The private Expo East simulator and standalone GS collection remain
separate HF assets. Paper-by-paper GS inventory reconciliation is not a release task. No source repository is made public,
no third-party data is uploaded. License scope is documented in [licensing](licensing.md).

## Checks before the first push

```bash
python scripts/check_release.py --repo /path/to/repository --history
python -m unittest discover -s tests -p 'test_*.py'
```

Run each App's documented tests/build and the UE release-tree checker as well.
The scanner's 50 MiB threshold is a conservative project policy, not a statement
of a hosting provider's limit. The optional history check inspects reachable Git
blobs as well; checking files does not remove any old committed secrets/assets.
