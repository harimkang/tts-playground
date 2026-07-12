# Third-party license notices

This file records the primary model/code licenses used by TTS Playground. It is a notice, not legal advice and not a replacement for the complete upstream terms. Review the linked license text before use or redistribution.

## Qwen3-TTS

Copyright 2026 Alibaba Qwen Team.

The `qwen-tts` code and the three Qwen3-TTS 1.7B model repositories used here are identified by their publishers as **Apache License 2.0 (Apache-2.0)**:

- Code and package metadata: <https://github.com/QwenLM/Qwen3-TTS> and <https://github.com/QwenLM/Qwen3-TTS/blob/main/pyproject.toml>
- CustomVoice model: <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice>
- VoiceDesign model: <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign>
- Base/voice-clone model: <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base>
- Apache License 2.0 text: <https://www.apache.org/licenses/LICENSE-2.0>

Preserve copyright, attribution, license, and NOTICE material required by Apache-2.0 when redistributing covered code, images, or model artifacts. Check each downloaded snapshot for additional upstream notices.

## Supertonic 3

Copyright 2026 Supertone Inc.

Supertonic separates the licenses for its code and model:

- The Supertonic project/sample code is released under the **MIT License**. Project and license: <https://github.com/supertone-inc/supertonic> and <https://github.com/supertone-inc/supertonic/blob/main/LICENSE>
- The accompanying `Supertone/supertonic-3` model is released under the **OpenRAIL-M License**. Model card and complete model license: <https://huggingface.co/Supertone/supertonic-3> and <https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE>

OpenRAIL-M includes use-based restrictions in addition to its grant of rights. Review and comply with the complete model license; do not treat the model as MIT-licensed merely because the surrounding sample code is MIT-licensed.

## Other dependencies

The complete frozen dependency set is recorded in `uv.lock`. Container base images, CUDA libraries, PyTorch, ONNX Runtime, and other transitive components retain their own licenses. Operators distributing the deployment must inventory the exact built artifacts and preserve all notices those licenses require.
