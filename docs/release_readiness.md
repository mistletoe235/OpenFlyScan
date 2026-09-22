# Release status

Status as of September 22, 2026. This page describes the private preview, not a
requirement to obtain project approval before running or training the code.

## Available

| Component | Current status |
| --- | --- |
| Source | Main, Android V4/V5, iOS and UE pushed to private `mistletoe235` repositories |
| Predictor | Trained head and eight-region inference example included in the main repository |
| Android packages | Signed V4 `0.3.1-v4` and V5 `0.1.1-v5` APKs in main/app Releases, with checksums and notices |
| iOS | Source and TestFlight request discussion available; no IPA distributed |
| Workstation | Pi3X, shared predictor and directional planner integrated; setup in [the service guide](workstation.md) |
| HIL simulator | Expo East Linux v0.1.0 package in the private HF dataset |

GitHub and HF permissions are separate. All repositories remain private;
creating a request in Discussions does not itself grant access or a TestFlight
invitation. The website's public download links have not been enabled.

## Outstanding items

- **Private contact:** a maintainer email or another private reporting channel
  has not been configured. Do not post sensitive details in Issues or Discussions.
- **TestFlight:** invitation delivery depends on the maintainer providing an
  eligible external-testing build; its availability has not been verified here.
- **Training data:** the downloadable teacher/cache bundle is not published.
  The included inference sample is not a training dataset. Use a prepared plan
  as described in [training](training_data.md); no cluster-specific approval is needed.
- **Hardware validation:** the Android packages passed build/signing checks,
  but this packaging pass did not include fresh phone or aircraft testing.
  Follow the [app safety procedure](../README.md#app-safety-notes), including simulator
  testing before flight.
- **UE distribution:** NanoGS source provenance and the final redistribution
  scope still need resolution. The existing HF v0.1.0 archive was not rebuilt
  after later source/notice changes; do not treat it as a build of current `main`.

Package tags, manifests and checksums identify the distributed binaries.
Documentation updates do not replace packages or move their tags. Historical
checks are recorded in [validation](validation.md); component license scopes
are in [licensing](licensing.md).
