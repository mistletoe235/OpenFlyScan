# UE / HIL simulator

Download the scene-inclusive Linux package from the
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
repository. The source and HF downloads are public. See the HF HIL guide for
phone setup, network directions and bench precautions.
