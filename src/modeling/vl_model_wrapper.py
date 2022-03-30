from typing import Iterable, Optional
from torch.nn.modules.loss import CrossEntropyLoss
from transformers import AutoConfig, AutoModel
from transformers.file_utils import ModelOutput
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput
from volta.encoders import *
from transformers.modeling_bart import BartDecoder, _prepare_bart_decoder_inputs
from src.modeling.model_full_bart import MGM_full_bart
import torch.nn as nn


def __get_model(config_file, model_path, pretrained_decoder=None):
    config = BertConfig.from_json_file(config_file)
    type_vocab_size = config.type_vocab_size
    config.type_vocab_size = 2
    model = BertForVLGeneration.from_pretrained(model_path, config=config,
                                                    from_hf=True, type_vocab_size=type_vocab_size, pretrained_decoder=pretrained_decoder)
    config.type_vocab_size = type_vocab_size
    return model

def get_vl_bert_model():
    config_file = 'config/vl-bert-base.json'
    checkpoint_path = 'data/in/volta_pretrained_models/vl-bert.bin'
    return __get_model(config_file, checkpoint_path)

def get_vil_bert_model():
    config_file = 'config/vil-bert-base.json'
    checkpoint_path = 'data/in/volta_pretrained_models/vil-bert.bin'
    return __get_model(config_file, checkpoint_path)

def get_lxmert_model():
    config_file = 'config/lxmert.json'
    checkpoint_path = 'data/in/volta_pretrained_models/lxmert.bin'
    return __get_model(config_file, checkpoint_path)

def get_vl_bert_pretrained_bart_model():
    config_file = 'config/vl-bert-base.json'
    checkpoint_path = 'data/in/volta_pretrained_models/vl-bert.bin'
    return __get_model(config_file, checkpoint_path, pretrained_decoder='facebook/bart-base')

def get_lxmert_pretrained_bart_model():
    config_file = 'config/lxmert.json'
    checkpoint_path = 'data/in/volta_pretrained_models/lxmert.bin'
    return __get_model(config_file, checkpoint_path, pretrained_decoder='facebook/bart-base')


def _make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer

class BertForVLGeneration(BertPreTrainedModel, GenerationMixin):
    def __init__(self, config, type_vocab_size, dropout_prob=0.1, pretrained_decoder=None):
        super(BertForVLGeneration, self).__init__(config)

        self.bert = BertModel(config)
        self.bert.embeddings.token_type_embeddings = \
        self._get_resized_embeddings(self.bert.embeddings.token_type_embeddings, type_vocab_size)
        self.shared_embeddings = self.bert.embeddings.word_embeddings
        self.shared_weights = self.shared_embeddings.weight
        self.dropout = nn.Dropout(dropout_prob)
        self.config = config
        
        if pretrained_decoder is not None :
            self.apply(self.init_weights)
            bart_config = AutoConfig.from_pretrained(pretrained_decoder)
            aux = AutoModel.from_pretrained(pretrained_decoder)
            self.decoder = aux.decoder
            self.bart_config = bart_config
            self.shared_embeddings = self.decoder.embed_tokens
            self.shared_weights = self.shared_embeddings.weight
            self.config.vocab_size = self.shared_embeddings.weight.shape[0]
        else:
            bart_config = AutoConfig.from_pretrained('facebook/bart-base')
            bart_config.vocab_size = self.shared_weights.size(0)
            self.bart_config = bart_config
            self.decoder = BartDecoder(bart_config, embed_tokens=self.shared_embeddings)
            self.apply(self.init_weights)
        self.fusion_method = config.fusion_method
        self.register_buffer("final_logits_bias", torch.zeros((1, self.shared_embeddings.num_embeddings)))
        print()

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.shared_embeddings)  # make it on the fly

    def get_encoder(self):
        return self.bert

    def forward(
        self,
        input_ids,
        visual_feats,
        visual_pos,
        batch_kinds=None,
        activate_img_features=None,
        token_type_ids=None,
        attention_mask=None,
        visual_attention_mask=None,
        output_all_encoded_layers=False,
        output_all_attention_masks=False,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels = None,
        target_indexes=None,
        **kwargs
    ):
                
        if encoder_outputs is None:
            encoded_layers_t, encoded_layers_v, *_ = self.bert(
                input_ids,
                visual_feats,
                visual_pos,
                token_type_ids,
                attention_mask,
                visual_attention_mask,
                output_all_encoded_layers=output_all_encoded_layers,
                output_all_attention_masks=output_all_attention_masks,
            )
            if batch_kinds[0] != 'img+img':
                encoder_outputs = torch.cat([encoded_layers_v, encoded_layers_t], 1)
                attention_mask = torch.cat([visual_attention_mask, attention_mask], 1)
            else:
                obj_features = encoded_layers_v[torch.arange(encoded_layers_v.shape[0]),target_indexes,:].unsqueeze(1)
                cls_ = encoded_layers_t[:, 0:1]
                # index -5 works for bert-base-cased tokenizer. Be aware!
                encoder_outputs = torch.cat([cls_, encoded_layers_v, encoded_layers_t[:,1:-5], obj_features, encoded_layers_t[:,-5:]], 1)
                attention_mask = torch.ones(*encoder_outputs.shape[:2]).to(encoder_outputs.device)
        
        decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.bart_config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.bert.embeddings.word_embeddings.weight.dtype,
            )
        decoder_out = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs,
                encoder_padding_mask=attention_mask,
                decoder_padding_mask=decoder_padding_mask,
                decoder_causal_mask=causal_mask,
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

        lm_logits = F.linear(
            decoder_out[0], self.shared_weights, bias=self.final_logits_bias)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=1)
            # TODO(SS): do we need to ignore pad tokens in labels?
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
        )

    def prepare_inputs_for_generation(self, input_ids, encoder_outputs, encoder_attention_mask, **kwargs):
        """
        Implement in subclasses of :class:`~transfomers.PreTrainedModel` for custom behavior to prepare inputs in the
        generate method.
        """
        d =  {"input_ids": input_ids, 'encoder_outputs': encoder_outputs.last_hidden_state, 'encoder_attention_mask':encoder_attention_mask, 
        'visual_feats':None, 'visual_pos':None}
        return d

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
                "Please use another model class (e.g. OpenAIGPTLMHeadModel, XLNetLMHeadModel, GPT2LMHeadModel, CTRLLMHeadModel, T5WithLMHeadModel, TransfoXLLMHeadModel, XLMWithLMHeadModel, BartForConditionalGeneration )"
            )
        old_conf = self.config
        self.config = self.bart_config
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
            
            encoded_layers_t, encoded_layers_v, *_ = encoder(
                input_ids,
                attention_mask=attention_mask,
                **encoder_kwargs,
            )
            encoder_outputs = torch.cat([encoded_layers_v, encoded_layers_t], 1)
            extended_attention_mask = torch.cat([torch.ones(*encoded_layers_v.shape[:-1]).to(encoder_outputs.device), attention_mask], 1)
            encoder_outputs = ModelOutput(extended_attention_mask=extended_attention_mask, last_hidden_state=encoder_outputs)

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
            model_kwargs['encoder_attention_mask'] = extended_attention_mask.repeat((num_beams, 1))


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
        self.config = old_conf
        return output

if __name__ =='__main__':
    get_vl_bert_pretrained_bart_model()
    get_lxmert_pretrained_bart_model()
