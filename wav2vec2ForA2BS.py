from typing import Optional, Tuple, Union
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC
from transformers.modeling_outputs import (
    CausalLMOutput,
)

class Wav2Vec2ForA2BS(Wav2Vec2ForCTC):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        """We inherit Wav2Vec2ForCTC but only return the outputs from wav2vec2
        Only by doing this we are able to use the pretrained model fine-tuned on chinese langeuage model.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs
        
    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
