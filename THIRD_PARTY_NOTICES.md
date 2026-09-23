# Third-party credits and licenses

This repository contains original recipe documentation and launch/validation scripts only. It does not copy, vendor, sublicense or redistribute third-party code or model weights.

- [IFM/K2-Horizon-MoVA-36B-A4B](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B): original BF16 model and learned routing biases, Apache License 2.0. IFM owns the model, data and training work. Respect its model card and license when downloading/extracting.
- [vcruz305/K2-Horizon-MoVA-36B-A4B-EXL3](https://huggingface.co/vcruz305/K2-Horizon-MoVA-36B-A4B-EXL3): EXL3 quant pack derived from IFM, Apache License 2.0 per model card. Weights are not redistributed here.
- [Turboderp ExLlamaV3](https://github.com/turboderp-org/exllamav3) and [Victor's K2 architecture fork](https://github.com/vcruz305/exllamav3): EXL3 trellis/quantization and inference library; MIT License, Copyright (c) 2025 Turboderp. Fork at pinned commit adds K2-Horizon support and the public routing-bias extractor. The fork's own notices apply when installed.
- [TabbyAPI](https://github.com/theroyallab/tabbyAPI): serving engine, GNU AGPL v3. Source is cloned at an exact commit and patched locally to pass the overlay. If operating a modified TabbyAPI over a network, comply with AGPL source-availability requirements. No TabbyAPI code is redistributed here.
- [Sixcat](https://github.com/vcruz305/sixcat-eval): unscored speed harness at v0.7.0 for the referenced baseline; its own license applies.

The recipe's MIT license covers **only the original contents of this repository**, not those separate components.
