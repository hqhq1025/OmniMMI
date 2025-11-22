import torch
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from decord import VideoReader, cpu

from base import ViLLMBaseModel

class Qwen25VL(ViLLMBaseModel):
    def __init__(self, model_args):
        super().__init__(model_args['model_path'], model_args['device'])
        assert(
            "model_path" in model_args
            and "device" in model_args
        )

        self.model_path = model_args['model_path']
        self.device = model_args['device']

        # Load model and processor
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=f"cuda:{self.device}"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        # Set model to eval mode
        self.model.eval()

    def generate(self, instruction, video_path):
        """
        Generate response for a video and instruction.

        Args:
            instruction: Text instruction/question
            video_path: Path to video file

        Returns:
            Generated text response
        """
        # Sample frames from video using decord
        max_frames = 32  # Qwen2-VL can handle multiple frames
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        # Prepare messages in Qwen2-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames,  # Pass frames directly
                    },
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        # Process the inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(f"cuda:{self.device}")

        # Generate response
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
            )

        # Trim the input tokens from generated output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode the output
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output_text.strip()
