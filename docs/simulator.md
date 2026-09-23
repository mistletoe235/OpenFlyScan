# UE / HIL simulator

**Download status:** the Expo East package is uploaded, but Hugging Face rejected
public access because the organization's public-storage quota is exceeded.
The archive is unchanged; public download is not available yet. The UE source
linked below is public and can be built independently.

Package location:
[OpenFlyScan HF dataset](https://huggingface.co/datasets/IPEC-COMMUNITY/openflyscan/tree/main/HIL-simulator).

Current package:
`HIL-simulator/linux/v0.1.0/OpenFlyScan-HIL-ExpoEast-Linux-v0.1.0.tar.zst`.

Extract the complete archive, then:

```bash
cd Linux/OpenFlySplatUE
./run_expo_east.sh
```

Expo East opens with the English HIL window waiting for a phone. Use
`./run_expo_east.sh --preview` for the software-only SimpleFlight preview.
The package also provides `convert_nanogs_ply.sh` for another GS scene.

The independent UE 5.5 source is hosted in the
[OpenFlyScan-UE repository](https://github.com/mistletoe235/OpenFlyScan-UE).
The renderer and AirSim remain there rather than being duplicated in the main
repository. See its `Docs/HIL_QUICKSTART.md` for phone setup, network directions
and bench precautions.
