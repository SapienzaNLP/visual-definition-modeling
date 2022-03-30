import logging
from typing import Iterable, List, Optional, Tuple

import torch
from src.data_preparation.multimodal_datasets import MultimodalTxtDataset
from src.modeling.components import (LxmertEncoderWrapper,
                                     bart_mod_encoder_forward)
from src.modeling.modeling_lxmert_mod import LxmertModelMod
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader
from transformers import (AutoModel, BartForConditionalGeneration,
                          BartTokenizer, BartTokenizerFast, LxmertTokenizer,
                          LxmertTokenizerFast)
from transformers.file_utils import ModelOutput
from transformers.generation_utils import GenerationMixin
from transformers.modeling_bart import BartEncoder

logger = logging.getLogger(__name__)


class MGM_full_bart(BartForConditionalGeneration):
    def __init__(
        self,
        config,
        multimodal_encoder="unc-nlp/lxmert-base-uncased",
        encoder_concat_img_features=True,
        img_dropout=0.5,
        lang_transformer_layers=-1,
        lang_transformer_heads=-1,
        img_transformer_layers=-1,
        img_transformer_heads=-1,
        joint_transformer_layers=-1,
        joint_transformer_heads=-1,
        encoder_feature_output=None,
        **unused,
    ):
        super().__init__(config)
        self.lxmert = LxmertEncoderWrapper(
            multimodal_encoder,
            concat_img_features=encoder_concat_img_features,
            img_dropout=img_dropout,
            lang_transformer_layers=lang_transformer_layers,
            lang_transformer_attention_heads=lang_transformer_heads,
            img_transformer_layers=img_transformer_layers,
            img_transformer_attention_heads=img_transformer_heads,
            joint_transfomer_layers=joint_transformer_layers,
            joint_transfomer_heads=joint_transformer_heads,
            encoder_output_features=encoder_feature_output,
        )
        self.encoder = self.lxmert
        BartEncoder.forward = bart_mod_encoder_forward
        self.model

    def forward(
        self,
        input_ids,
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
        **kwargs,
    ):
        assert (
            "token_type_ids" in kwargs
            and "visual_feats" in kwargs
            and "visual_pos" in kwargs
            and "visual_attention_mask" in kwargs
            and "batch_kinds" in kwargs
            and "activate_img_features" in kwargs
        ) or encoder_outputs is not None
        if encoder_outputs is None:
            (
                token_type_ids,
                visual_feats,
                visual_pos,
                visual_attention_mask,
                batch_kinds,
                activate_img_features,
            ) = (
                kwargs[x]
                for x in [
                    "token_type_ids",
                    "visual_feats",
                    "visual_pos",
                    "visual_attention_mask",
                    "batch_kinds",
                    "activate_img_features",
                ]
            )
            dict_out = self.lxmert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                visual_feats=visual_feats,
                visual_pos=visual_pos,
                visual_attention_mask=visual_attention_mask,
                return_dict=True,
                activate_img_features=activate_img_features,
            )
            lxmert_outputs = dict_out.last_hidden_state
            attention_mask = dict_out.extended_attention_mask
        else:
            # attention_mask = encoder_outputs.extended_attention_mask
            encoder_outputs = (
                (encoder_outputs,)
                if type(encoder_outputs) == torch.Tensor
                else (encoder_outputs.last_hidden_state,)
            )
            lxmert_outputs = encoder_outputs[0]

        return super().forward(
            lxmert_outputs,
            attention_mask=attention_mask,
            encoder_outputs=None,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def get_img_dropout_target(self):
        return self.encoder.img_dropout_val_target

    def get_encoder(self):
        return self.lxmert

    def set_img_dropout(self, val):
        self.encoder.img_dropout_val = val

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        encoder_kwargs=None,
        **model_kwargs,
    ) -> torch.LongTensor:
        # We cannot generate if the model does not have a LM head
        if self.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
            )

        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = (
            early_stopping if early_stopping is not None else self.config.early_stopping
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = (
            temperature if temperature is not None else self.config.temperature
        )
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = (
            repetition_penalty
            if repetition_penalty is not None
            else self.config.repetition_penalty
        )
        bos_token_id = (
            bos_token_id if bos_token_id is not None else self.config.bos_token_id
        )
        pad_token_id = (
            pad_token_id if pad_token_id is not None else self.config.pad_token_id
        )
        eos_token_id = (
            eos_token_id if eos_token_id is not None else self.config.eos_token_id
        )
        length_penalty = (
            length_penalty if length_penalty is not None else self.config.length_penalty
        )
        no_repeat_ngram_size = (
            no_repeat_ngram_size
            if no_repeat_ngram_size is not None
            else self.config.no_repeat_ngram_size
        )
        bad_words_ids = (
            bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        )
        num_return_sequences = (
            num_return_sequences
            if num_return_sequences is not None
            else self.config.num_return_sequences
        )
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.config.decoder_start_token_id
        )

        if input_ids is not None:
            # overriden by the input batch_size
            batch_size = input_ids.shape[0]
        else:
            batch_size = 1

        assert (
            isinstance(max_length, int) and max_length > 0
        ), "`max_length` should be a strictly positive integer."
        assert (
            isinstance(min_length, int) and min_length >= 0
        ), "`min_length` should be a positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
        assert (
            isinstance(num_beams, int) and num_beams > 0
        ), "`num_beams` should be a strictly positive integer."
        assert temperature > 0, "`temperature` should be strictly positive."
        assert (
            isinstance(top_k, int) and top_k >= 0
        ), "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, "`length_penalty` should be strictly positive."
        assert (
            isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
        ), "`no_repeat_ngram_size` should be a positive integer."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictly positive integer."
        assert (
            bad_words_ids is None
            or isinstance(bad_words_ids, list)
            and isinstance(bad_words_ids[0], list)
        ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1),
                bos_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
        else:
            assert (
                input_ids.dim() == 2
            ), "Input prompt should be of shape (batch_size, sequence length)."

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert (
                    num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # beam_search greedy generation conditions
                assert (
                    num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        if (
            (attention_mask is None)
            and (pad_token_id is not None)
            and (pad_token_id in input_ids)
        ):
            attention_mask = input_ids.ne(pad_token_id).long()
        elif attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # set pad_token_id to eos_token_id if not set. Important that this is done after
        # attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(
                "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(
                    eos_token_id
                )
            )
            pad_token_id = eos_token_id

        # vocab size
        if hasattr(self.config, "vocab_size"):
            vocab_size = self.config.vocab_size
        elif (
            self.config.is_encoder_decoder
            and hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "vocab_size")
        ):
            vocab_size = self.config.decoder.vocab_size
        else:
            raise ValueError(
                "either self.config.vocab_size or self.config.decoder.vocab_size needs to be defined"
            )

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                # see if BOS token can be used for decoder_start_token_id
                if bos_token_id is not None:
                    decoder_start_token_id = bos_token_id
                elif (
                    hasattr(self.config, "decoder")
                    and hasattr(self.config.decoder, "bos_token_id")
                    and self.config.decoder.bos_token_id is not None
                ):
                    decoder_start_token_id = self.config.decoder.bos_token_id
                else:
                    raise ValueError(
                        "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
                    )

            assert hasattr(
                self, "get_encoder"
            ), "{} should have a 'get_encoder' function defined".format(self)
            assert callable(self.get_encoder), "{} should be a method".format(
                self.get_encoder
            )

            # get encoder and store encoder outputs
            encoder = self.get_encoder()
            encoder_outputs: ModelOutput = encoder(
                input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                **encoder_kwargs,
            )

        # Expand input ids if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams > 1:
            attention_mask = encoder_outputs.extended_attention_mask
            input_ids_len = input_ids.shape[-1]
            input_ids = input_ids.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len
            )
            attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size,
                effective_batch_mult * num_beams,
                encoder_outputs.last_hidden_state.shape[1],
            )

            input_ids = input_ids.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            attention_mask = attention_mask.contiguous().view(
                effective_batch_size * num_beams,
                encoder_outputs.last_hidden_state.shape[1],
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

        if self.config.is_encoder_decoder:
            # create empty decoder input_ids
            input_ids = torch.full(
                (effective_batch_size * num_beams, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
            cur_len = 1

            assert (
                batch_size == encoder_outputs.last_hidden_state.shape[0]
            ), f"expected encoder_outputs.last_hidden_state to have 1st dimension bs={batch_size}, got {encoder_outputs.last_hidden_state.shape[0]} "

            # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
            expanded_batch_idxs = (
                torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .to(input_ids.device)
            )

            # expand encoder_outputs
            encoder_outputs[
                "last_hidden_state"
            ] = encoder_outputs.last_hidden_state.index_select(0, expanded_batch_idxs)

            # save encoder_outputs in `model_kwargs`
            model_kwargs["encoder_outputs"] = encoder_outputs

        else:
            cur_len = input_ids.shape[-1]

        assert (
            cur_len < max_length
        ), f"The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` or `config.max_length = ...`"

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                early_stopping=early_stopping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                num_beams=num_beams,
                vocab_size=vocab_size,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_kwargs=model_kwargs,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_kwargs=model_kwargs,
            )

        return output


if __name__ == "__main__":
    bart_name = "facebook/bart-base"
    lxmert_name = "unc-nlp/lxmert-base-uncased"
    encoder_tokenizer = LxmertTokenizer.from_pretrained(lxmert_name)
    decoder_tokenizer = BartTokenizer.from_pretrained(bart_name)

    mgm_bart = MGM_full_bart.from_pretrained(bart_name, multimodal_encoder=lxmert_name)

    dataset = MultimodalTxtDataset(
        encoder_tokenizer,
        decoder_tokenizer,
        "<d>",
        "</d>",
        "data/in/semeval2007.data.dm.txt",
        # "data/in/semcor.data.dm.subset.dev.txt",
        "data/in/babelpic.frcnn.semcor.all.npz",
        limit_sentences=-1,
    )
    loader = DataLoader(dataset, batch_size=1, collate_fn=dataset.get_batch_fun())
    generation_params = dict(
        num_beams=4,
        min_length=5,
        max_length=15,
        temperature=1,
        repetition_penalty=2,
        length_penalty=1.5,
        no_repeat_ngram_size=2,
    )
    mgm_bart.eval()
    """
    self, input_ids=None,
                visual_feats=None,
                visual_pos=None,
                attention_mask=None,
                token_type_ids=None,
                visual_attention_mask=None,
                return_dict=True,
                activate_img_features=True
    """
    for batch in loader:
        beam_output = mgm_bart.generate(
            batch["input_ids"],
            **generation_params,
            decoder_start_token_id=decoder_tokenizer.bos_token_id,
            decoder_kwargs={},
            encoder_kwargs={
                "visual_feats": batch["visual_feats"],
                "visual_pos": batch["visual_pos"],
                "visual_attention_mask": batch["visual_attention_mask"],
                "activate_img_features": batch["activate_img_features"],
                "token_type_ids": batch["token_type_ids"],
            },
        )
        print("Output:\n" + 100 * "-")
        print(encoder_tokenizer.decode(batch["input_ids"][0]))
        print(decoder_tokenizer.decode(beam_output[0]))
        print(decoder_tokenizer.decode(batch["labels"][0]).strip())
