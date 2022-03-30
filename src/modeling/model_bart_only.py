import warnings

from torch.nn import Parameter, Module, CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoModel, BartForConditionalGeneration, BartTokenizerFast, LxmertTokenizerFast, \
    LxmertTokenizer, BartTokenizer
import torch
from transformers.generation_utils import GenerationMixin
from transformers.modeling_bart import BartEncoder, shift_tokens_right
from transformers.modeling_outputs import Seq2SeqLMOutput

from src.data_preparation.multimodal_datasets import MultimodalTxtDataset
from src.modeling.components import LxmertEncoderWrapper, bart_mod_encoder_forward
from src.modeling.modeling_lxmert_mod import LxmertModelMod
import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BartOnlyWrapper(BartForConditionalGeneration):
    def __init__(self, config, multimodal_encoder="unc-nlp/lxmert-base-uncased",
                 encoder_concat_img_features=True, img_dropout=0.5,
                 lang_transformer_layers=-1, lang_transformer_heads=-1,
                 img_transformer_layers=-1, img_transformer_heads=-1,
                 joint_transformer_layers=-1, joint_transformer_heads=-1,
                 encoder_feature_output=None,
                 **unused):
        super().__init__(config)

    def forward(self, input_ids,
                attention_mask=None,
                encoder_outputs=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                past_key_values=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **unused, ):
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = unused.pop("lm_labels")
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_past_key_values")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=1)
            # TODO(SS): do we need to ignore pad tokens in labels?
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def get_img_dropout_target(self):
        return -1

    # def get_encoder(self):
    #     return self.lxmert

    def set_img_dropout(self, val):
        pass


if __name__ == "__main__":
    bart_name = "facebook/bart-base"
    lxmert_name = "unc-nlp/lxmert-base-uncased"
    encoder_tokenizer = LxmertTokenizer.from_pretrained(lxmert_name)
    decoder_tokenizer = BartTokenizer.from_pretrained(bart_name)

    mgm_bart = BartOnlyWrapper.from_pretrained(bart_name)

    dataset = MultimodalTxtDataset(encoder_tokenizer, decoder_tokenizer, "<d>", "</d>",
                                   "data/in/semeval2007.data.dm.txt",
                                   # "data/in/semcor.data.dm.subset.dev.txt",
                                   "data/in/babelpic.frcnn.semcor.all.npz", limit_sentences=-1
                                   )
    loader = DataLoader(dataset, batch_size=1, collate_fn=dataset.get_batch_fun())
    generation_params = dict(num_beams=4,
                             min_length=5,
                             max_length=15,
                             temperature=1,
                             repetition_penalty=2,
                             length_penalty=1.5,
                             no_repeat_ngram_size=2)
    mgm_bart.eval()
    for batch in loader:
        beam_output = mgm_bart.generate(
            batch["input_ids"],
            **generation_params,
            decoder_start_token_id=decoder_tokenizer.bos_token_id,
            decoder_kwargs={},
            encoder_kwargs={"visual_feats": batch["visual_feats"], "visual_pos": batch["visual_pos"],
                            "visual_attention_mask": batch["visual_attention_mask"]
                            }

        )
        print("Output:\n" + 100 * '-')
        print(encoder_tokenizer.decode(batch["input_ids"][0]))
        print(decoder_tokenizer.decode(beam_output[0]))
        print(decoder_tokenizer.decode(batch["labels"][0]).strip())
