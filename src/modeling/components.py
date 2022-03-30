from dataclasses import dataclass
import logging
import random
import sys
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, TransformerEncoder, TransformerEncoderLayer
from transformers.modeling_bart import BartEncoder
from transformers.modeling_outputs import BaseModelOutput

from src.modeling.modeling_lxmert_mod import LxmertModelMod

logger = logging.getLogger(__name__)


def swish(x):
    return x * torch.sigmoid(x)

def prepare_encoder_input(batch):
    encoder_kwargs={
                        "visual_feats": batch["visual_feats"],
                        #"visual_attention_mask": batch["visual_attention_mask"],
                        "activate_img_features": batch["activate_img_features"],
                        "target_indexes": batch["target_indexes"],
                        "insert_img_objects": batch["insert_img_objects"],
                    }
    return encoder_kwargs

def get_empty_visual_features(batch_size, device):
    features = torch.zeros(batch_size, 1, 2048).to(device)
    pos = (torch.zeros(batch_size, 1, 4) - 1).to(device)
    return features, pos


class TrainingParams(dict):
    def __init__(self, other_dict=None, **kwargs):
        super().__init__()
        if other_dict is not None:
            self.update(other_dict)
        self.update(kwargs)

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(TrainingParams, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(TrainingParams, self).__delitem__(key)
        del self.__dict__[key]

@dataclass
class BartWrapperModelOutput(BaseModelOutput):
    attention_mask: Optional[torch.LongTensor] = None
    


class BartEncoderWrapper(Module):

    def __init__(self, bart_encoder, visual_feats_hidden_size=-1):
        super().__init__()
        self.bart_encoder = bart_encoder
        self.rand_vals = torch.rand(1000).tolist()
        if visual_feats_hidden_size > 0:
            self.feature_mapper = Linear(visual_feats_hidden_size, self.bart_encoder.embed_tokens.embedding_dim)


    #def get_img_dropout_val(self):
    #    return self.lxmert_encoder.img_dropout_val

    def forward(self,
                input_ids, attention_mask, visual_feats=None, target_indexes=None, output_attentions=None, 
                output_hidden_states=None, activate_img_features=None, insert_img_objects=None, return_dict=None, 
                train_img=torch.no_grad(), train_bart=torch.no_grad()):

        #if self.training:
        #    if len(self.rand_vals) == 0:
        #        self.rand_vals = torch.rand(1000).tolist()
        #    rand_val = self.rand_vals.pop()
        #    if self.get_img_dropout_val() > rand_val:
        #        activate_img_features = False

        if activate_img_features:
            # adding 1 to the shape of the image outputs so as to allow to add a </s> extra token
            # at the end
            with train_img:
                visual_feats = self.feature_mapper(visual_feats)
            attention_mask = torch.cat([torch.ones(visual_feats.shape[0], visual_feats.shape[1]+ 1).to(attention_mask.device),
                                        attention_mask], -1)
            if insert_img_objects:
                # adjust in order to add the attention mask for the obj feature
                attention_mask = torch.cat([torch.ones(attention_mask.shape[0], 1, dtype=torch.bool).to(attention_mask.device), attention_mask], axis=-1)
           
            with train_bart:
                encoder_outputs = self.bart_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    crossmod_embeds=visual_feats,
                    target_indexes=target_indexes,
                    insert_img_objects=insert_img_objects,
                )

        else:
            with train_bart:
                encoder_outputs = self.bart_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

        if not return_dict:
            return encoder_outputs[0], attention_mask

        return BartWrapperModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                attention_mask= attention_mask
            )


class LxmertEncoderWrapper(Module):
    def __init__(self, model_name,
                 img_transformer_layers=2,
                 img_transformer_attention_heads=2,
                 lang_transformer_attention_heads=2,
                 lang_transformer_layers=2,
                 joint_transfomer_layers=1,
                 joint_transfomer_heads=4,
                 encoder_output_features=None,
                 concat_img_features=False,
                 img_dropout=0.0,
                 output_only_img_features=False):
        super().__init__()
        self.output_only_img_features = output_only_img_features
        self.concat_img_features = concat_img_features
        self.encoder = LxmertModelMod.from_pretrained(model_name)
        self.img_dropout_val = img_dropout
        self.img_dropout_val_target = img_dropout
        self.rand_vals = torch.rand(1000).tolist()
        self.lang_transformer = self.init_transformer(lang_transformer_layers, lang_transformer_attention_heads,
                                                      self.encoder.config.hidden_size)
        self.img_transformer = self.init_transformer(img_transformer_layers, img_transformer_attention_heads,
                                                     self.encoder.config.hidden_size)
        self.joint_transfomer = self.init_transformer(joint_transfomer_layers, joint_transfomer_heads,
                                                     self.encoder.config.hidden_size)
        if encoder_output_features is not None:
            self.feature_mapper = Linear(self.encoder.config.hidden_size, encoder_output_features)
        else:
            self.feature_mapper = None

    def init_transformer(self, layers, attention_heads, size):
        if layers < 0:
            return None
        return TransformerEncoder(
            TransformerEncoderLayer(size, attention_heads), layers)
        # hidden_size_map = Linear(self.encoder.config.hidden_size, lang_feature_map)

    def forward(self, input_ids=None,
                visual_feats=None,
                visual_pos=None,
                attention_mask=None,
                token_type_ids=None,
                visual_attention_mask=None,
                return_dict=True,
                activate_img_features=True):
        if input_ids is None and visual_feats is None:
            logger.error("at least one among input_ids and visual_feats have to be set")
            sys.exit(1)
        concat_img_features = self.concat_img_features
        '''
        if self.training:
            if len(self.rand_vals) == 0:
                self.rand_vals = torch.rand(1000).tolist()
            rand_val = self.rand_vals.pop()
            if self.img_dropout_val > rand_val:
                visual_feats, visual_pos = visual_feats * 0.0, (visual_pos * 0.0) - 1
                activate_img_features = False
        '''

        if visual_feats is None:
            concat_img_features = False
            visual_feats, visual_pos = get_empty_visual_features(input_ids.size(0), input_ids.device)
        dict_out = self.encoder(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                visual_feats=visual_feats,
                                visual_pos=visual_pos,
                                visual_attention_mask=visual_attention_mask,
                                return_dict=return_dict,
                                activate_visual_feats=activate_img_features)
        lang_encoded = dict_out.language_output
        visual_encoded = dict_out.vision_output

        if self.lang_transformer is not None:
            lang_encoded = self.lang_transformer(lang_encoded)

        text_attention_mask = attention_mask
        if concat_img_features and activate_img_features:
            if self.img_transformer is not None:
                visual_encoded = self.img_transformer(visual_encoded)
            encoded_input = torch.cat([visual_encoded, lang_encoded], 1)
            if text_attention_mask is None:
                text_attention_mask = torch.ones_like(input_ids).to(encoded_input.device)
            if visual_attention_mask is None:
                extended_attention_mask = torch.cat(
                    [torch.ones(visual_encoded.shape[:-1]).to(text_attention_mask.device), text_attention_mask], -1)
            else:
                extended_attention_mask = torch.cat([visual_attention_mask, text_attention_mask], -1)
        elif self.output_only_img_features:
            encoded_input = visual_encoded
            extended_attention_mask = torch.ones(visual_encoded.shape[:-1]).to(text_attention_mask.device)
        else:
            encoded_input = lang_encoded
            extended_attention_mask = text_attention_mask
        if self.joint_transfomer is not None:
            encoded_input = self.joint_transfomer(encoded_input)

        if self.feature_mapper is not None:
            encoded_input = self.feature_mapper(encoded_input)
        dict_out["last_hidden_state"] = encoded_input
        dict_out["extended_attention_mask"] = extended_attention_mask
        return dict_out


# import math
# from sentence_transformers import util
# def mbrr_rescoring(sentence_embedder, generations):
#     embeddings = sentence_embedder.encode([g["output"] for g in generations], convert_to_tensor=True,
#     device="cuda")
#     similarities = util.pytorch_cos_sim(embeddings.cuda(), embeddings.cuda())
#     for i in range(len(generations)): ## can be replaced with a matrix multiplication ( similarities x generations[:,"log_prob"])
#         scores = list()
#         for j in range(0, len(generations)):
#             if j == i:
#                 continue
#             gjp = generations[j]["sentence_probability"]
#             ij_score = similarities[i][j].item()
#             new_score = (1 - ij_score) * gjp
#             scores.append(new_score)
#         mbrr_score = sum(scores)
#         generations[i]["mbrr_score"] = mbrr_score

#     return sorted(generations, key=lambda elem: elem["mbrr_score"])


def invert_mask(attention_mask):
    """Turns 1->0, 0->1, False->True, True-> False"""
    assert attention_mask.dim() == 2
    return attention_mask.eq(0)


# class BartEncoderMod(BartEncoder):
def bart_mod_encoder_forward(self,
                             input_ids,
                             attention_mask=None,
                             output_attentions=False,
                             output_hidden_states=False,
                             return_dict=False,
                             crossmod_embeds=None,
                             target_indexes=None,
                             insert_img_objects=None,
                             ):
                             
    """
    Args:
        input_ids (FloatTensor): tokens in the source language of shape
            `(batch, src_len, embed_dim)`
        attention_mask (torch.LongTensor): indicating which indices are padding tokens.
    Returns:
        BaseModelOutput or Tuple comprised of:
            - **x** (Tensor): the last encoder layer's output of
              shape `(src_len, batch, embed_dim)`
            - **encoder_states** (tuple(torch.FloatTensor)): all intermediate
              hidden states of shape `(src_len, batch, embed_dim)`.
              Only populated if *output_hidden_states:* is True.
            - **all_attentions** (tuple(torch.FloatTensor)): Attention weights for each layer.
            During training might not be of length n_layers because of layer dropout.
    """
    # check attention mask and invert
    #attention_mask = torch.cat([torch.ones(crossmod_embeds.shape[:-1]).to(attention_mask.device), attention_mask], -1)
    if attention_mask is not None:
        attention_mask = invert_mask(attention_mask)
    inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    start_embeds = inputs_embeds[:,0].unsqueeze(1)
    end_embeds = inputs_embeds[input_ids==2].unsqueeze(1)
    if crossmod_embeds is not None and not insert_img_objects:
        inputs_embeds = torch.cat([start_embeds, crossmod_embeds, end_embeds, inputs_embeds[:,1:]], 1)
    elif crossmod_embeds is not None and insert_img_objects:
        obj_features = crossmod_embeds[torch.arange(crossmod_embeds.shape[0]),target_indexes,:].unsqueeze(1)
        inputs_embeds = torch.cat([start_embeds, crossmod_embeds, end_embeds, inputs_embeds[:,1:-4], obj_features, inputs_embeds[:,-4:]], 1)
        # inputs_embeds = torch.cat([crossmod_embeds, inputs_embeds[:,:-4,:], 
        # crossmod_embeds[torch.arange(crossmod_embeds.shape[0]),target_indexes,:].unsqueeze(1), inputs_embeds[:,-4:,:]], 1) * self.embed_scale
    

    embed_pos = self.embed_positions(inputs_embeds)
    x = inputs_embeds + embed_pos
    x = self.layernorm_embedding(x)
    x = F.dropout(x, p=self.dropout, training=self.training)

    # B x T x C -> T x B x C
    x = x.transpose(0, 1)

    encoder_states = [] if output_hidden_states else None
    all_attentions = () if output_attentions else None
    for encoder_layer in self.layers:
        if output_hidden_states:
            encoder_states.append(x)
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        dropout_probability = random.uniform(0, 1)
        if self.training and (dropout_probability < self.layerdrop):  # skip the layer
            attn = None
        else:
            x, attn = encoder_layer(x, attention_mask, output_attentions=output_attentions)

        if output_attentions:
            all_attentions = all_attentions + (attn,)

    if self.layer_norm:
        x = self.layer_norm(x)
    if output_hidden_states:
        encoder_states.append(x)
        # T x B x C -> B x T x C
        encoder_states = tuple(hidden_state.transpose(0, 1) for hidden_state in encoder_states)

    # T x B x C -> B x T x C
    x = x.transpose(0, 1)

    if not return_dict:
        return tuple(v for v in [x, encoder_states, all_attentions] if v is not None)
    return BaseModelOutput(last_hidden_state=x, hidden_states=encoder_states, attentions=all_attentions)

# class BartModelMod(BartModel):
#     def __init__(self, config: BartConfig):
#         super().__init__(config)
#         self.encoder.forward = bart_mod_encoder_forward
