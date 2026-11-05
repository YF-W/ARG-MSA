import os
import torch
import torch.nn as nn
from transformers import (
    BertModel, BertTokenizer,
    RobertaModel, RobertaTokenizer,
)

__all__ = ['BertTextEncoder']

TRANSFORMERS_MAP = {
    'bert': (BertModel, BertTokenizer),
    'roberta': (RobertaModel, RobertaTokenizer),
}


def _resolve_local_path(pretrained: str) -> (str, bool):
    """
    Map pretrained name to local path when available.
    Returns (path_or_name, local_files_only_flag).
    """
    is_windows = os.name == 'nt'
    path_map = {
        # English
        'bert-base-uncased': r'D:\pretrained\bert-base-uncased' if is_windows else '/public/home/sixinheyi/.cache/huggingface/transformers/bert-base-uncased-local',
        # Chinese
        'bert-base-chinese': r'D:\pretrained\bert-ch' if is_windows else '/public/home/sixinheyi/.cache/huggingface/transformers/bert_ch',
    }
    target = path_map.get(pretrained, pretrained)
    use_local = os.path.isdir(target)
    return target, use_local


class BertTextEncoder(nn.Module):
    def __init__(self, use_finetune: bool = False, transformers: str = 'bert', pretrained: str = 'bert-base-uncased'):
        super().__init__()
        tokenizer_class = TRANSFORMERS_MAP[transformers][1]
        model_class = TRANSFORMERS_MAP[transformers][0]

        resolved, local_only = _resolve_local_path(pretrained)
        self.tokenizer = tokenizer_class.from_pretrained(resolved, local_files_only=local_only)
        self.model = model_class.from_pretrained(resolved, local_files_only=local_only)
        self.use_finetune = use_finetune

    def forward(self, text: torch.Tensor):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, attention_mask, token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:, 0, :].long(), text[:, 1, :].float(), text[:, 2, :].long()
        if self.use_finetune:
            last_hidden_states = self.model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids
            )[0]
        else:
            with torch.no_grad():
                last_hidden_states = self.model(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    token_type_ids=segment_ids
                )[0]
        return last_hidden_states
