"""CPU parity of the DeepSeek-V4.1 token-space strategy with the legacy processor; no weights.

serving   served output == legacy PIL backend: IDs, items, offsets, patches, n_vit_h/n_vit_w;
          PNG, JPEG and RGBA, padded, stretched, upscaled and token-capped resizes, and two
          adjacent images -> test_serving_output_matches_legacy_processor
hashes    transposed solid images have equal patches but different grids and spans
          -> test_transposed_images_get_different_hashes
turns     a later turn adds one image behind the expanded prefix
          -> test_partial_expansion_adds_the_new_turns_image
towers    a text-only checkpoint rejects images; an nvJPEG decoder is rejected at startup
          -> test_unsupported_inputs_are_rejected
trainer   the strategy module imports without serving modules
          -> test_strategy_imports_without_serving_modules
"""

import asyncio
import io
import subprocess
import sys
import unittest

import msgspec
import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from sglang.srt.configs.deepseek_v41 import DeepseekV41Config
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.multimodal.processors.deepseek_v41 import DeepseekV41ImageProcessor
from sglang.srt.multimodal.token_space_mm_process.models.deepseek_v41 import (
    DeepseekV41TokenSpaceProcessStrategy,
    DeepseekV41TokenSpaceServingSpec,
)
from sglang.srt.multimodal.token_space_mm_process.sglang import (
    TokenSpaceSGLangProcessor,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

ARCH = "DeepseekV4ForCausalLM"
IMAGE = 129264
VOCAB = {"[UNK]": 0, "hi": 1, "there": 2, "<image>": IMAGE}

DeepseekV41TokenSpaceProcessor = (
    TokenSpaceSGLangProcessor.for_strategy_and_serving_spec(
        DeepseekV41TokenSpaceProcessStrategy, DeepseekV41TokenSpaceServingSpec
    )
)


def _tokenizer():
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]")),
        unk_token="[UNK]",
    )


def _config(*, vision=True):
    # Patches of 4 px in 8 px token cells keep the grids small; the planner's branches
    # depend on the ratios, not on the checkpoint's 14 px patches.
    return DeepseekV41Config(
        architectures=[ARCH],
        vision_n_layers=1 if vision else 0,
        image_token_id=IMAGE,
        vision_patch_size=4,
        vision_downsample_ratio=2,
        vision_max_n_token=40,
        vision_min_pixels=256,
        vision_max_wh_ratio=4,
    )


def _image(width, height, seed, format, mode="RGB"):
    channels = len(mode)
    pixels = np.random.RandomState(seed).rand(height, width, channels) * 255
    buffer = io.BytesIO()
    Image.fromarray(pixels.astype("uint8"), mode).save(buffer, format=format)
    return buffer.getvalue()


def _solid_png(width, height):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 90)).save(buffer, format="PNG")
    return buffer.getvalue()


class TestDeepseekV41TokenSpace(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        self.server_args = ServerArgs(
            model_path="dummy",
            model_impl="sglang",
            mm_process_config={},
            mm_feature_transport="cpu",
            mm_preprocess_cache_size_mb=0,
            # Pins legacy to its PIL backend, the one the strategy reproduces.
            disable_fast_image_processor=True,
            mm_processor_worker_num=1,
            enable_token_space_processor=True,
        )
        publish(self.server_args, role="test")
        self.tokenizer = _tokenizer()
        self.images = [
            _image(16, 16, 1, "PNG"),  # exact grid
            _image(24, 10, 2, "JPEG"),  # upscaled to vision_min_pixels, then padded
            _image(20, 20, 3, "PNG", mode="RGBA"),  # alpha dropped
            _image(64, 8, 4, "PNG"),  # wider than vision_max_wh_ratio: stretched
            _image(80, 60, 5, "PNG"),  # over vision_max_n_token: shrunk
        ]

    def _processors(self, config=None):
        config = config or _config()
        legacy = DeepseekV41ImageProcessor(
            config, self.server_args, self.tokenizer, None
        )
        self.addCleanup(legacy.shutdown)
        served = DeepseekV41TokenSpaceProcessor(
            config, self.server_args, self.tokenizer, None
        )
        self.addCleanup(served.shutdown)
        return legacy, served

    def _served_output(self, served, input_ids, images, **request):
        return asyncio.run(
            served.process_mm_data_async(
                image_data=list(images),
                input_text=input_ids,
                request_obj=GenerateReqInput(input_ids=input_ids, **request),
            )
        )

    def _legacy_output(self, legacy, input_ids, images):
        return asyncio.run(
            legacy.process_mm_data_async(
                image_data=list(images),
                input_text=input_ids,
                request_obj=GenerateReqInput(input_ids=input_ids),
            )
        )

    def assert_same_media(self, served, legacy):
        self.assertEqual(served.input_ids, legacy.input_ids)
        self.assertEqual(served.im_token_id, legacy.im_token_id)
        self.assertEqual(len(served.mm_items), len(legacy.mm_items))
        for item, reference in zip(served.mm_items, legacy.mm_items):
            self.assertEqual(item.modality, reference.modality)
            self.assertEqual(item.offsets, reference.offsets)
            self.assertEqual(item.feature.dtype, reference.feature.dtype)
            self.assertTrue(torch.equal(item.feature, reference.feature))
            item, reference = (msgspec.structs.asdict(i) for i in (item, reference))
            self.assertEqual(
                item["model_specific_data"], reference["model_specific_data"]
            )

    def test_serving_output_matches_legacy_processor(self):
        """Every resize branch of the planner, with the last two images adjacent."""
        legacy, served = self._processors()
        input_ids = [1, IMAGE, 2, IMAGE, 1, IMAGE, 2, IMAGE, IMAGE]
        served_output = self._served_output(served, input_ids, self.images)
        legacy_output = self._legacy_output(legacy, input_ids, self.images)
        self.assert_same_media(served_output, legacy_output)
        # The token-capped image still fits vision_max_n_token.
        start, end = served_output.mm_items[-1].offsets[0]
        self.assertLessEqual(end - start + 1, 40)

    def test_transposed_images_get_different_hashes(self):
        """A solid 32x16 and 16x32 image patchify to equal tensors; only their grids differ.

        A hash of the patches alone would let one reuse the other's embedding, whose span
        has a different length.
        """
        _, served = self._processors()
        output = self._served_output(
            served,
            [IMAGE, 1, IMAGE],
            [_solid_png(32, 16), _solid_png(16, 32)],
        )
        wide, tall = output.mm_items
        self.assertTrue(torch.equal(wide.feature, tall.feature))
        self.assertNotEqual(wide.hash, tall.hash)

    def test_partial_expansion_adds_the_new_turns_image(self):
        """The second turn resends every image; only the placeholder after the prefix expands."""
        legacy, served = self._processors()
        first_turn = self._served_output(served, [1, IMAGE, 2], self.images[:1])
        second_turn_ids = first_turn.input_ids + [1, IMAGE]
        second_turn = self._served_output(
            served,
            second_turn_ids,
            self.images[:2],
            mm_token_expansion_start_len=len(first_turn.input_ids),
        )
        legacy_output = self._legacy_output(
            legacy, [1, IMAGE, 2, 1, IMAGE], self.images[:2]
        )
        self.assert_same_media(second_turn, legacy_output)

    def test_unsupported_inputs_are_rejected(self):
        """Neither reaches the model: it has no ViT, or patchify_image would get a GPU tensor."""
        served = DeepseekV41TokenSpaceProcessor(
            _config(vision=False), self.server_args, self.tokenizer, None
        )
        self.addCleanup(served.shutdown)
        with self.assertRaisesRegex(ValueError, "no vision tower"):
            self._served_output(served, [IMAGE], self.images[:1])
        with self.assertRaisesRegex(ValueError, "image_decoder must be 'pil'"):
            DeepseekV41TokenSpaceProcessStrategy(
                _config(),
                self.tokenizer,
                mm_process_config={"image_decoder": "nvjpeg"},
            )

    def test_strategy_imports_without_serving_modules(self):
        """Trainers import the strategy; the serving spec defers its serving imports."""
        loaded = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys\n"
                "import sglang.srt.multimodal.token_space_mm_process.models.deepseek_v41\n"
                "print(sorted(m for m in sys.modules if m.startswith(("
                "'sglang.srt.managers', 'sglang.srt.models', 'sglang.srt.server_args', "
                "'sglang.srt.multimodal.processors.base_processor'))))",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.assertEqual(loaded, "[]")


if __name__ == "__main__":
    unittest.main()
