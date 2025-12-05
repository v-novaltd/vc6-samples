# VC-6 Samples
VC-6 Python SDK – Sample Scripts License: BSD-3-Clause-Clear (see LICENSE) Ownership: © 2014–2025 V-Nova International Limited Distribution: Made available by V-Nova Limited, a wholly owned subsidiary of V-Nova International Limited.

Note: This repository contains sample Python scripts only. It does not include VC-6 SDK wheels, installers, or license keys

Reference samples for V-Nova's SMPTE VC-6 codec python module.
These samples demonstrate how to encode, decode, and validate VC-6's functionality and performance.


## VC-6 Python SDK

## Requirements for samples
   - Python 3.10+ recommended
   - CUDA 12.9 (For CUDA encoder and decoder)

```cmd
pip install -r requirements.txt
```

## Samples

### 1. Encoding:
   - VC-6 CPU Encoder.
   - VC-6 OpenCL GPU Encoder.
   - VC-6 CUDA GPU Encoder.

### 2. Decoding:
   - VC-6 CPU Decoder.
   - VC-6 OpenCL GPU Decoder.
   - VC-6 CUDA GPU Decoder.
   - VC-6 Decoder with partial fetch.
   - VC-6 Decoder with Region of Interest Decoding.
   - VC-6 CUDA GPU Decoder with CUDA Memory to resize with Nvidia DALI.


## Contributing (temporarily closed)
We’re not accepting external contributions right now. We may open issues and pull requests in the future—watch or star this repo for updates.
If you plan to contribute later, include this header at the top of each new source file:
```cmd
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2014-2025 V-Nova International Limited
```

## Security
If you believe you’ve found a vulnerability in these samples, email ai@v-nova.com with details and a reproducible example if possible. Please do not open public issues for sensitive reports.

## Support / questions 
For now, please contact ai@v-nova.com. We’ll enable GitHub Issues when contributions open.

## Configure VC-6 Codec
The first time the codec is imported there will be a EULA message that needs to be accepted which will fetch the license for VC-6 Codec automatically. This is a one time process, any subsequent run should directly invoke the codec. You can expect to output for the first run to be as follows.

```cmd
$ python decode/opencl_decoder.py


V-Nova VC-6 Python SDK EULA
=====================================================

V-Nova VC-6 SDK - Evaluation License
Copyright (c) 2014-2025, V-Nova International Limited.
Published by V-Nova Limited. All rights reserved.

1. Definitions.
"Software" means the VC-6 SDK Python wheel package and any associated binaries, libraries, headers, tools and documentation delivered by V-Nova Limited.
"Licensee" means the person or entity installing or using the Software.

2. Grant. 
Subject to this License, V-Nova Limited grants Licensee a personal, non-exclusive, non-transferable, non-sublicensable, time-limited evaluation licence to use the Software in object-code form only, for internal testing, integration and demonstration solely to assess suitability for commercial deployment, for fourteen (14) days from first activation (the "Evaluation Period"). Commercial or production use is not permitted and requires a separate written agreement with V-Nova.

3. Technical time-limit
The Software may include technical measures that count down from first activation and automatically disable functionality when the Evaluation Period ends. Licensee must not circumvent any technical limit, time check, rate-limit or security.

4. Restrictions
Licensee shall not: (a) use the Software in production or for any commercial or revenue-generating purpose; (b) copy, distribute, host, resell or provide the Software to third parties; (c) reverse-engineer, decompile or disassemble, or attempt to derive source code or underlying ideas, except where this restriction is prohibited by applicable law; (d) remove or alter notices or identifiers; (e) publish or disclose benchmarks or performance results without V-Nova's prior written consent; (f) use the Software to create a competing product; (g) bypass evaluation limits.

5. Ownership; no patent licences; feedback.
The Software is licensed, not sold. All intellectual-property rights are owned by V-Nova International Limited. Except for the limited rights granted here, no other rights are granted, including no patent licences. If Licensee provides feedback, Licensee grants V-Nova a worldwide, irrevocable, perpetual, royalty-free licence to use and incorporate it without restriction.

6. Confidentiality
Non-public features, SDK interfaces, documentation and performance data are V-Nova confidential information. Licensee shall protect them with at least reasonable care and use them only as permitted here.


7. Telemetry
The Software may collect limited operational data (e.g., IP address, license/activation identifiers, SDK version, picture format, resolution, encode/decode counts, timestamps) for security and service operation. No media content is collected. Processing is described in PRIVACY.md included with the Software and referenced in the GitHub Release.

8. Warranty disclaimer. 
TO THE MAXIMUM EXTENT PERMITTED BY LAW, THE SOFTWARE IS PROVIDED "AS IS" AND "AS AVAILABLE", WITH ALL FAULTS, AND WITHOUT ANY WARRANTIES OR CONDITIONS OF ANY KIND, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING WITHOUT LIMITATION WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT.

9. Liability.
TO THE MAXIMUM EXTENT PERMITTED BY LAW, V-NOVA SHALL NOT BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY OR CONSEQUENTIAL DAMAGES, OR LOSS OF PROFITS, REVENUE, DATA OR GOODWILL. V-Nova's aggregate liability for direct damages is limited to GBP 1,000. Nothing limits liability that cannot be limited by law.

10. Termination
This License automatically terminates at the end of the Evaluation Period, or immediately upon breach. Upon termination or expiry, Licensee must stop using the Software and destroy all copies and keys. Sections 3-12 survive termination.

11. Export & compliance
 Licensee will comply with applicable export-control, sanctions and technology-control laws and will not use or export the Software in violation of such laws.

12. Governing law
This License and any non-contractual obligations arising out of or in connection with it are governed by the laws of England and Wales, and the courts of England have exclusive jurisdiction.

13. Entire agreement; changes. This License is the entire agreement for evaluation use of the Software and supersedes prior understandings on that subject. No waiver or modification is binding unless in writing signed by V-Nova.

For extensions or a commercial licence, contact licensing@v-nova.com.


Do you accept the VC-6 SDK EULA? [y/N]: 
```

## Run Samples
```cmd
$ python decode/decode_resize_cuda_memory_dali.py --help


usage: decode_resize_cuda_memory_dali.py [--help] [-w MAXWIDTH] [-h MAXHEIGHT] [-b BATCH] [-l LOQ] [-rw RESIZEWIDTH] [-rh RESIZEHEIGHT] -s SOURCE -d DESTINATION_DIR

Decode all VC-6 images in a directory (or single VC-6 image) to Raw RGB format and resize using Nvidia DALI.

options:
  --help                Show this help message and exit
  -w MAXWIDTH, --maxwidth MAXWIDTH
                        Maximum width of images (default: 2048)
  -h MAXHEIGHT, --maxheight MAXHEIGHT
                        Maximum height of images (default: 2048)
  -b BATCH, --batch BATCH
                        Batch size for decode (default: 10)
  -l LOQ, --loq LOQ     Level of Quality LOQ (default: 0)
  -rw RESIZEWIDTH, --resizewidth RESIZEWIDTH
                        Resize Width (default: 224)
  -rh RESIZEHEIGHT, --resizeheight RESIZEHEIGHT
                        Resize height (default: 224)
  -s SOURCE, --source SOURCE
                        Path to source directory containing images, or a single image file
  -d DESTINATION_DIR, --destination-dir DESTINATION_DIR
                        Directory to write decoded files
```

# VC-6 Documentation
 
## Installation 
 The only requirement is Python 3.8 or above.
 Install the package via the provided wheel compatible with your platform and architecture.

```cmd
pip install vc6[cu12]
pip install vc6[opencl]
```

## Usage

Core codec functionality is inside the `codec` module that can be imported as follow:

```python
from `vnova.vc6_opencl` import `codec` as vc6codec
```

or, if you have installed the CUDA codec:

```python
from `vnova.vc6_cu12` import `codec` as vc6codec
```

Then, you can create codecs and start transcoding. For complete examples, see the provided sample codes:

```python
# setup encoder and decoder instances
    encoder = vc6codec.EncoderSync(
        1920, 1080, vc6codec.CodecBackendType.CPU, vc6codec.PictureFormat.RGB_8, vc6.ImageMemoryType.CPU)
    encoder.set_generic_preset(vc6codec.EncoderGenericPreset.LOSSLESS)
    # Using double resolution to demonstrate reconfiguration later
    decoder = vc6codec.DecoderSync(1920, 1080, vc6codec.CodecBackendType.CPU, vc6codec.PictureFormat.RGB_8, vc6codec.ImageMemoryType.CPU)
    # encode file to memory
    encoded_image = encoder.read("example_1920x1080_rgb8.rgb")
    # decode memory to memory
    decoded_image = decoder.write(encoded_image.memoryview, "reconstruction_example_1920x1080_rgb8.rgb")
```

### GPU memory output
When using our CUDA package (`vc6_cu12`), the decoder output can be in device memory.
To use this feature, create the decoder with specifying `GPU_DEVICE` as the output memory type.
With that, the output images will have `__cuda_array_interface__` and can be used with other libraries like CuPy, PyTorch and nvImageCodec.

```python
    import cupy
    # setup GPU decoder instances with CUDA device output
    decoder = vc6codec.DecoderSync(1920, 1080, vc6codec.CodecBackendType.CPU, vc6codec.PictureFormat.RGB_8, vc6codec.ImageMemoryType.CUDA_DEVICE)
    # decode from file
    decoded_image = decoder.read("example_1920x1080_rgb8.vc6")
    # Make a cupy array from decoded image, download to cpu and write to file
    cuarray = cupy.asarray(decoded_image)
    with open("reconstruction_example_1920x1080_rgb8.rgb") as decoded_file: 
        decoded_file.write(cuarray.get(), "reconstruction_example_1920x1080_rgb8.rgb")
```

Both for sync and async decoders, accessing `__cuda_array_interface__` is blocking and implicitly waits on the result to be ready in the image.

The `__cuda_array_interface__` always contains one-dimensional data of unsigned 8-bit type like the CPU version.
Adjusting dimensions (or the type in case of 10-bit formats) is up to the user.  

### Environment Variables:

Environment variables `OCL_BIN_LOC` and `OCL_DEVICE` can be set to define the GPU binary cache location and hint for selecting target GPU respectively.
For more details refer to [VC6-SDK documentation](https://docs.v-nova.com/technologies/smpte.vc-6/).

```cmd
export OCL_BIN_LOC=./tmp/clbin
export OCL_DEVICE=nvidia
```
Variable `CUDA_BIN_LOC` serves the same purpose for the CUDA version:

```cmd
export CUDA_BIN_LOC=./tmp/clbin
```
