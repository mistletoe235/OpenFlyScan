# Five-repository preparation — September 21, 2026

[English](preparation_20260921.md) · [Chinese reference](preparation_20260921.zh-CN.md)

This is a dated preparation record. All five source repositories were subsequently
pushed privately under `mistletoe235`; see [current repository links](../components.json).
The source/asset boundaries below are unchanged.

## Repository boundaries

The five source repositories are the OpenFlyScan main repository, Android V4,
Android V5, iOS and UE. The online workstation service belongs in the main
repository, not a sixth repository. The existing website and HF dataset are reused.

## Completed in this pass

| Item | Result |
| --- | --- |
| V5 continuous-recapture negotiation | New sessions explicitly request schema 14 when selected; schema 13 remains the default |
| Android route preview | Unapproved missions open an independent read-only diagram, not an execution task |
| Three-client service interfaces | V4/V5/iOS passed session/mission checks against the new service; no flight was executed |
| Main CPU example | Synthetic inputs check loading/inference without external data or a GPU |
| Real inference example | Approximately 5 MB of head weights, eight-region features and reference outputs included with source |
| Dependency recovery | Pinned GeoFF3D revision and patch; all eight restored modified files matched byte-for-byte |
| Training entry | Data template/path resolution, direct single-/multi-GPU training and resume; cluster approval/code-snapshot prerequisites removed |
| UE release preparation | External name updated to OpenFlyScan UE while retaining compatible internal names; source/package version correspondence documented |
| Asset inventory | 35 existing GS candidates retained as an upload index, about 20.96 GiB; paper-by-paper reconciliation removed from the task list |

After adding weights, examples and licenses, 127 main-repository tests passed.
Earlier V4/V5 targeted runs passed 31/49 tests and Debug builds; 28 iOS cloud tests
passed. See the [validation record](validation.md) for scope.

All five repositories ran source-release checks. Bounded credential-pattern and
large-file scans of current files/reachable history found no matches; the UE
release-tree check passed. Main original code, the head and sample use Apache-2.0;
backbone weights and third-party assets retain separate terms. See [licensing](licensing.md).

## Publication items recorded at the time

- A private security contact and the five GitHub owners/names still needed confirmation.
- Changes needed review/commit and remotes needed configuration; that preparation pass did not create remotes, commit or push. The later private push is recorded in the repository links above.
- Head weights and the small sample ship in the main repository. GS assets and simulator packages remain on HF; paper-by-paper GS reconciliation is not a release task.
- The teacher/cache training bundle was not published; templates and tools alone are not the complete training data.
- App signing, valid SDK keys and hardware acceptance remain separate from source publication.
- UE distribution packages should correspond to fixed source revisions and recorded checksums. The existing Expo East package's provenance was not rewritten.

See [release readiness](release_readiness.md), the asset index at
`../configs/assets.release.json`, and head/sample checksums at
`../configs/quality_predictor.release.json`.
