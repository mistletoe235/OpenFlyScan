# Validation — 2026-09-21

## Local repository

- Python 3.10, PyTorch 2.7.1, NumPy 2.2.6 and SciPy 1.15.3 on macOS, CPU.
- `python -m unittest discover -s tests -p 'test_*.py'`: **127 tests passed** after bundling the head/sample and completing license packaging.
- The Python wheel builds successfully and contains the project license,
  third-party notices and retained upstream license texts.
- Direct-start regression tests reach backbone initialization without a code
  snapshot, authorization file or preceding diagnostic run; GPU/backbone loading
  is mocked. CPU synthetic training checks optimizer updates and exact resume.
  No full GPU retraining was launched for this change.
- Training, prediction and evaluation command-line `--help` entry points passed.
- Local Markdown targets, Python syntax and JSON parsing passed.
- A synthetic checkpoint saved before renaming loads strictly through the new
  API: parameter keys/shapes match, and predictions are bitwise identical
  (maximum absolute difference: **0.0**).
- Source inventory paths resolve. Recorded hashes identify the original archive;
  public module names and imports have since changed. See `naming.md`.
- A limited credential-pattern scan found no matching private keys or HF/GitHub
  tokens. This is not a complete publication security review.

## Workstation replay and deployment

- The paper Quality Predictor checkpoint loads strictly: **1,252,725 parameters**.
  Its original frozen implementation and the renamed implementation produce
  bitwise-identical CPU predictions on the same real feature inputs.
- A CUDA replay processes 30 pre-reacquisition Expo West images through Pi3X,
  feature extraction and scoring. Explicit geometry inputs have shape
  `[64, 30, 8]`, and regional descriptors have shape `[64, 32]`.
- A separate HTTP replay uploads those images, finalizes the session and returns
  a preview PLY, quality targets and a schema 14 reacquisition mission.
- The current Android V5 native decoder and validator accept both the frozen
  239-capture reference export and the HTTP-generated 6-capture schema 14 mission;
  serialization round trips preserve their contents. The reference export also
  preserves capture poses across schema 13 and schema 14.
- Production port **55000** now runs `openflyscan.server.http` from the main
  repository. All four historical sessions and their PLY endpoints remain
  accessible, and the existing access token is unchanged. Creating and cancelling
  empty sessions succeeds for both mission schemas.

No aircraft commands were issued. The 30-image replay is an integration test,
not a new full-scene benchmark. Full retraining and App/UE builds were not run.
The local Git repository still has no remote, commit or public upload.

## Release preparation follow-up

- Android V4: 31 focused cloud/mission tests passed; Android V5: 49 passed.
  Both include an opt-in live-service check, and Debug builds passed.
- iOS: 28 cloud/upload tests passed, including the live-service check against
  a newly generated schema 13 mission. A historical legacy mission was correctly
  rejected for inconsistent capture metadata; its validator was not weakened.
- A new 30-image HTTP replay generated the schema 13 reference on the deployed
  service. Client tests created/cancelled empty sessions and read that result;
  they did not execute a flight.
- The GeoFF3D runtime patch restores all eight modified files byte-for-byte from
  the recorded base revision. The archived COLMAP reader is included unchanged.
- The inference-only exported checkpoint preserves CPU predictions exactly on
  eight real regional inputs. The same example runs on macOS, with maximum
  Linux/macOS score difference of approximately `4.77e-7`.
