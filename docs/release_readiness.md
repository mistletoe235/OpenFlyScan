# Release status

Status as of September 23, 2026. All five source repositories and the Expo East
HF package are public. This page records availability, not an approval process
for use or training.

## Available

| Component | Current status |
| --- | --- |
| Source | Main, Android V4/V5, iOS and UE public under `mistletoe235` |
| Predictor | Trained head and eight-region inference example included in the main repository |
| Android packages | Signed V4 `0.3.1-v4` and V5 `0.1.1-v5` APKs in main/app Releases, with checksums and notices |
| iOS | Source and TestFlight request discussion available; no IPA distributed |
| Workstation | Pi3X, shared predictor and directional planner integrated; setup in [the service guide](workstation.md) |
| HIL simulator | Expo East Linux v0.1.0 publicly available on HF |

The website links to source, Android downloads and simulator downloads.
iOS installation still requires a TestFlight invitation; source access does not
itself issue an invitation.

Questions and requests go through [GitHub Discussions](https://github.com/mistletoe235/OpenFlyScan/discussions).
No separate maintainer email is required. Do not post sensitive details there.

## Outstanding items

- **TestFlight:** invitation delivery depends on the maintainer providing an
  eligible external-testing build; its availability has not been verified here.
- **Training data:** the downloadable teacher/cache bundle is not published.
  The included inference sample is not a training dataset. Use a prepared plan
  as described in [training](training_data.md); no cluster-specific approval is needed.
- **Hardware validation:** the Android packages passed build/signing checks,
  but this packaging pass did not include fresh phone or aircraft testing.
  Follow the [app safety procedure](../README.md#app-safety-notes), including simulator
  testing before flight.

Package tags, manifests and checksums identify the distributed binaries.
Documentation updates do not replace packages or move their tags. Historical
checks are recorded in [validation](validation.md); component license scopes
are in [licensing](licensing.md).
