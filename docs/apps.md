# Mobile applications

OpenFlyScan uses the OpenFly Go applications for capture and reconstruction feedback.

| Application | Pinned SDK / reference aircraft | Mission schemas | Source link |
| --- | --- | --- | --- |
| Android V4 | DJI MSDK 4.16.4 / Mini 2 | 1–14 | [OpenFlyGo-Android-V4](https://github.com/mistletoe235/OpenFlyGo-Android-V4) |
| Android V5 | DJI MSDK 5.18.0 / Mini 4 Pro | 1–14 | [OpenFlyGo-Android-V5](https://github.com/mistletoe235/OpenFlyGo-Android-V5) |
| iOS | DJI MSDK 4.16.2 / Mini 2 | 1–14 | [OpenFlyGo-iOS](https://github.com/mistletoe235/OpenFlyGo-iOS) |

The apps are separate public repositories under `mistletoe235`; their source
links are also registered in `components.json`.
These source links are separate from the signed-package release links below.
Build keys and signing credentials must be supplied privately, never stored here.
DJI HIL instructions are provided with the simulator; platform features are not
assumed to be identical across Android V4/V5 and iOS.

Android V4/V5 and iOS cloud-session dialogs explicitly request schema 14 when continuous
recapture is selected; the default remains schema 13. Android's existing-session
browser can show an unapproved route in a separate read-only diagram, without
activating it. Import and execution approval remain distinct from downloading
or viewing the route. See [the preparation report](preparation_20260921.md).

## Installation packages

Use the [main release](https://github.com/mistletoe235/OpenFlyScan/releases/tag/mobile-20260925) or the matching app release:
[Android V4](https://github.com/mistletoe235/OpenFlyGo-Android-V4/releases/tag/v0.3.5-v4),
[Android V5](https://github.com/mistletoe235/OpenFlyGo-Android-V5/releases/tag/v0.1.7-v5).
Both locations host the same publicly downloadable signed APK bytes. Android
packages target arm64 devices running Android 7.0 or later.
Check the SDK/aircraft table before installing, preserve missions when updating,
and do not bypass signature mismatches by deleting app data. Installation does not
validate an aircraft/firmware combination or start a flight.

### iOS TestFlight

[Join the iOS beta on TestFlight](https://testflight.apple.com/join/br5vTV92)

Open the link on your iPhone and follow the instructions to install TestFlight
and join the beta. No GitHub access request is needed. Availability depends on
the testing capacity and device requirements shown on TestFlight.

The app is not currently listed on the App Store. Check aircraft compatibility
and the flight-safety notes before use. For help, use the
[project discussion](https://github.com/mistletoe235/OpenFlyScan/discussions/1);
do not post passwords or verification codes.

## SDK support and version differences

- [DJI official supported products / platforms](https://developer.dji.com/mobile-sdk/)
- [V4 product support](https://developer.dji.com/mobile-sdk/documentation/introduction/product_introduction.html#supported-products)
- [Android V4 pinned release](https://github.com/dji-sdk/Mobile-SDK-Android/tree/V4.16.4)
- [iOS V4 pinned release](https://github.com/dji-sdk/Mobile-SDK-iOS/tree/v4.16.2)
- [V5 supported products](https://github.com/dji-sdk/Mobile-SDK-Android-V5#what-is-dji-mobile-sdk-v5)

SDK support is not project acceptance for every aircraft, payload, camera mode or
remote controller. Mini 2 and Mini 4 Pro are reference hardware with project usage
records, not a new all-feature flight acceptance for each release. Check the pinned
SDK, platform, firmware and exact camera against each client's README and
`docs/CAMERA_PROFILE_COMPATIBILITY.md`. Do not infer iOS support from Android support.

V4/iOS Mini 2 routes use app-side control: keep the app foreground and connected;
they are not upload-and-disconnect onboard missions. V5 additionally has DJI KMZ
execution. Schema 13 stops for capture; V5 schema 14 continuous recapture is
experimental, requires the DJI KMZ path and may still stop at turns/boundaries.
Android V4 and iOS implement schema 14 with bounded app-side Virtual Stick guidance and
photo acknowledgements, preserving stopped capture at ineligible points. It requires
the updated 2026-09-22 builds and device acceptance; default stopped missions remain schema 13.
Never enable continuous flight by merely changing a schema number.
Public apps exclude MNN/VLN and private model runtimes, not the route/cloud workflow.

## Basic workflow

1. Install the correct signed Android client from Releases, join the iOS TestFlight beta,
   or build from source with your own DJI key and signing settings.
   Read its `README.md` for setup, camera checks, survey region and height/overlap settings.
2. Rehearse the complete workflow in the app's built-in simulator before real
   flight. For an additional hardware-loop check, follow the client's
   `docs/HIL_QUICKSTART.md` and the [simulator guide](simulator.md) with propellers
   removed. Xcode Mock is not DJI flight-controller HIL.
3. Deploy the [workstation service](workstation.md) once. Enter its phone-reachable
   HTTPS root URL and bearer token in the app; no phone-to-SSH connection is needed.
4. Create an upload session, check image FOV and takeoff ASL, upload survey-trigger
   frames or geotagged historical photos, then explicitly finalize reconstruction.
   Without takeoff ASL the service can reconstruct but does not export a flight mission.
5. Refresh the session, view PLY, download the compatible proposed mission and preview
   it. Use the client's `docs/CLOUD_ROUTE_WORKFLOW.md` for import and preflight.
   Current server exports are review-only, not automatic flight authorization.
6. Check camera geometry, coordinate/height datum, path, battery and approval state
   before explicit execution. New capture rounds need their own collection/submission;
   downloading a cloud mission does not start a flight.

The new iOS upload section supports disk-backed queues, manual resume, historical
JPEG/PNG from Files/Photos and explicit finalize/retry/cancel. See its
`docs/CLOUD_UPLOAD.md`. Live images are fresh survey-trigger downlink frames, not
onboard SD originals; historical images must retain original GPS and ASL. Background
pauses uploads, and HIL/Mock frames are not mixed into real upload sessions. Existing
session browsing remains read-only. These additions require an updated iOS build;
source/test validation is not real-device/GPU reconstruction acceptance.
