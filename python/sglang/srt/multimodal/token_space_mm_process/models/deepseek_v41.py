"""DeepSeek-V4.1 token-space strategy: one ``image_token_id`` placeholder per image.

Every position of an image's span carries ``image_token_id``; the model tells IMAGE_START,
IMAGE, IMAGE_NEW_LINE and IMAGE_END apart by position. So the placeholder expands into a
run of itself, and the caller's other IDs, which Engram hashes, pass through unchanged.
"""

from types import MappingProxyType

import torch

from sglang.srt.multimodal.deepseek_v41_image_processing import (
    num_image_tokens,
    patchify_image,
)
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.token_space_mm_process.process_strategy import (
    TokenSpaceProcessStrategy,
)
from sglang.srt.multimodal.token_space_mm_process.serving_spec import (
    TokenSpaceServingSpec,
)


def _has_vision_tower(hf_config) -> bool:
    # The condition DeepseekV4ForCausalLM builds its ViT under; language_model_only is optional.
    return (
        hf_config.model_type == "deepseek_v41"
        and hf_config.vision_n_layers > 0
        and not getattr(hf_config, "language_model_only", False)
    )


class DeepseekV41TokenSpaceProcessStrategy(TokenSpaceProcessStrategy):
    """ViT patches resized and padded with PIL, as the legacy processor's PIL backend does."""

    models = ("DeepseekV4ForCausalLM",)
    mm_item_fields = MappingProxyType(
        {
            "pixel_values": Modality.IMAGE,
            "n_vit_h": Modality.IMAGE,
            "n_vit_w": Modality.IMAGE,
        }
    )
    mm_feature_fields = frozenset({"pixel_values"})
    # As legacy: patchify_image reads a PIL image, and nvJPEG pixels differ from PIL's.
    gpu_image_decode = False

    # Adapted from DeepseekV41ImageProcessor.__init__, without the Rust and GPU backend switch.
    def __init__(self, hf_config, processor, *, mm_process_config=None):
        super().__init__(hf_config, processor, mm_process_config=mm_process_config)
        if self.gpu_image_decode is not False:
            raise ValueError(
                "DeepSeek-V4.1 patchifies PIL images; mm_process_config image_decoder "
                "must be 'pil'"
            )
        self._has_vision_tower = _has_vision_tower(hf_config)
        self.mm_token_expansion_params = {
            "image_token_id": (
                hf_config.image_token_id if self._has_vision_tower else None
            ),
        }

    def load_image(self, source):
        if not self._has_vision_tower:
            raise ValueError(
                "This checkpoint has no vision tower, so image input is not supported"
            )
        return super().load_image(source)

    def process_image(self, image, processor, preprocess_kwargs, **kwargs):
        patches, n_vit_h, n_vit_w, _, _ = patchify_image(image, self.hf_config)
        return {
            "pixel_values": patches,
            "n_vit_h": torch.tensor([n_vit_h]),
            "n_vit_w": torch.tensor([n_vit_w]),
        }

    # Adapted from DeepseekV41ImageProcessor.process_mm_data_async; the token grid is the
    # one DeepseekV4ForCausalLM.get_image_feature builds the span from.
    def get_mm_token_expansion_spec(self, media_features, *, encode_expansion_text):
        image_token_id = self.mm_token_expansion_params["image_token_id"]
        if image_token_id is None:
            return {}
        ratio = self.hf_config.vision_downsample_ratio
        grids = (
            zip(media_features["n_vit_h"].tolist(), media_features["n_vit_w"].tolist())
            if "n_vit_h" in media_features
            else ()
        )
        return {
            image_token_id: [
                [image_token_id]
                * num_image_tokens((h + ratio - 1) // ratio, (w + ratio - 1) // ratio)
                for h, w in grids
            ]
        }


class DeepseekV41TokenSpaceServingSpec(TokenSpaceServingSpec):
    models = DeepseekV41TokenSpaceProcessStrategy.models

    def build_mm_tokens(self, processor):
        # Serving-only: the trainer imports this module without the serving base.
        from sglang.srt.multimodal.processors.base_processor import (
            MultimodalSpecialTokens,
        )

        if not _has_vision_tower(self.hf_config):
            return None
        image_token_id = self.hf_config.image_token_id
        return MultimodalSpecialTokens(
            image_token=processor.convert_ids_to_tokens(image_token_id),
            image_token_id=image_token_id,
        ).build(processor)

    def split_mm_items(self, mm_items):
        """One item per image with its own patch grid, as legacy: the model encodes item by item."""
        from sglang.srt.managers.schedule_batch import MultimodalDataItem

        split = []
        for item in mm_items:
            heights = torch.as_tensor(item.n_vit_h).tolist()
            widths = torch.as_tensor(item.n_vit_w).tolist()
            patches = item.feature.split([h * w for h, w in zip(heights, widths)])
            for offsets, feature, h, w in zip(item.offsets, patches, heights, widths):
                split.append(
                    MultimodalDataItem(
                        modality=item.modality,
                        feature=feature,
                        offsets=[offsets],
                        model_specific_data={"n_vit_h": h, "n_vit_w": w},
                    )
                )
        return split
