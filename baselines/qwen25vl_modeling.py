import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np
from PIL import Image

from base import ViLLMBaseModel


class Qwen25VL(ViLLMBaseModel):
    def __init__(self, model_args):
        super().__init__(model_args['model_path'], model_args['device'])
        assert(
            "model_path" in model_args
            and "device" in model_args
        )

        # Load model with Flash Attention 2 for efficiency
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args['model_path'],
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=f"cuda:{model_args['device']}"
        )

        # Load processor
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            model_args['model_path'],
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels
        )

        self.device = model_args['device']
        self.model_args = model_args

    def _load_video(self, video_path, max_frames=16):
        """Load and sample video frames uniformly"""
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)

        # Uniform sampling
        if total_frame_num > max_frames:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
        else:
            frame_idx = list(range(total_frame_num))

        frames = vr.get_batch(frame_idx).asnumpy()

        # Convert to list of PIL images
        pil_images = []
        for frame in frames:
            pil_images.append(Image.fromarray(frame))

        return pil_images

    def _prepare_messages(self, instruction, video_path):
        """Prepare messages for Qwen2.5VL"""
        images = self._load_video(video_path)

        # Create messages in Qwen2.5VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": images},
                    {"type": "text", "text": instruction}
                ]
            }
        ]

        return messages

    def generate(self, instruction, video_path):
        """
        Generate response for a video QA task

        Args:
            instruction: Text question/instruction
            video_path: Path to video file

        Returns:
            Generated text response
        """
        # Prepare messages
        messages = self._prepare_messages(instruction, video_path)

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(f"cuda:{self.device}")

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.001,
                top_k=1,
                do_sample=True,
                num_beams=1
            )

        # Remove input tokens from generation
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode output
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output_text.strip()
