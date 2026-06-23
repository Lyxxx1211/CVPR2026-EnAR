

import warnings
from typing import List, Optional, Tuple, Union

import torch.utils.checkpoint
from transformers.cache_utils import DynamicCache
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import LlamaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM, Qwen3MoeForCausalLM

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit_lx import InternVisionModel, has_flash_attn

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "InternVisionModel",
        "Qwen3DecoderLayer",
    ]


    _tp_plan = ''

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            architecture: str = config.llm_config.architectures[0]
            if architecture == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif architecture == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            elif architecture == 'Qwen3MoeForCausalLM':
                self.language_model = Qwen3MoeForCausalLM(config.llm_config)
            elif architecture == 'Qwen3ForCausalLM':
                self.language_model = Qwen3ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{architecture} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)


        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = min(selected.sum(), vit_embeds.size(0))
            input_embeds[selected][:n_token] = input_embeds[selected][:n_token] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()

        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds


    def extract_feature_lyx(self, pixel_values):


        assert self.select_layer == -1, "Visual token pruning is not implemented for select_layer not equal to -1"
        res = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
            output_attn_weights=True)
        attn_weights = res["attn_weights"]

        vit_embeds = res["last_hidden_state"]
        if pixel_values.shape[0] > 1:


            attn_weights_avg = torch.mean(attn_weights, dim=2)


            cls_to_patch_attention = attn_weights_avg[:, 5, 0, 1:]

            cls_to_patch_attention = cls_to_patch_attention.view(cls_to_patch_attention.shape[0], 32, 32)
            cls_to_patch_attention = cls_to_patch_attention.unfold(1, 2, 2).unfold(2, 2, 2)
            cls_to_patch_attention = cls_to_patch_attention.sum(dim=(3, 4))
            cls_to_patch_attention = cls_to_patch_attention.view(cls_to_patch_attention.shape[0], 256)

            cls_to_patch_attention = (cls_to_patch_attention - cls_to_patch_attention.min(dim=1, keepdim=True)[0]) / (cls_to_patch_attention.max(dim=1, keepdim=True)[0] - cls_to_patch_attention.min(dim=1, keepdim=True)[0])

            thumbnail_cls_attention = cls_to_patch_attention[-1, :]
            tile_cls_attention = cls_to_patch_attention[:-1, :]


        else:


            attn_weights_avg = torch.mean(attn_weights, dim=2)

            cls_to_patch_attention = attn_weights_avg[:, -1, 0, 1:]

            cls_to_patch_attention = cls_to_patch_attention.view(cls_to_patch_attention.shape[0], 32, 32)
            cls_to_patch_attention = cls_to_patch_attention.unfold(1, 2, 2).unfold(2, 2, 2)
            cls_to_patch_attention = cls_to_patch_attention.sum(dim=(3, 4))
            cls_to_patch_attention = cls_to_patch_attention.view(cls_to_patch_attention.shape[0], 256)

            cls_to_patch_attention = (cls_to_patch_attention - cls_to_patch_attention.min(dim=1, keepdim=True)[0]) / (cls_to_patch_attention.max(dim=1, keepdim=True)[0] - cls_to_patch_attention.min(dim=1, keepdim=True)[0])

            thumbnail_cls_attention = cls_to_patch_attention[-1, :]
            tile_cls_attention = None


        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)

        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)

        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)

        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds, thumbnail_cls_attention, tile_cls_attention


    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, pixel_values_recons, question,indices_diffusion, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')


        query_textonly = query.replace('<image>', '', 1)

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)


        model_inputs_textonly = tokenizer(query_textonly, return_tensors='pt')
        input_ids_textonly = model_inputs_textonly['input_ids'].to(self.device)
        attention_mask_textonly = model_inputs_textonly['attention_mask'].to(self.device)

        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            pixel_values_recons=pixel_values_recons,
            input_ids=input_ids,
            attention_mask=attention_mask,
            attention_mask_textonly=attention_mask_textonly,
            indices_diffusion=indices_diffusion,
            input_ids_textonly=input_ids_textonly,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            pixel_values_recons: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            indices_diffusion: Optional[torch.LongTensor] = None,
            output_hidden_states: Optional[bool] = None,
            input_ids_textonly: Optional[torch.FloatTensor] = None,
            attention_mask_textonly: Optional[torch.LongTensor] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds, thumbnail_cls_attention, tile_cls_attention = self.extract_feature_lyx(pixel_values)
                vit_embeds_useless,thumbnail_cls_attention_recons, tile_cls_attention_recons = self.extract_feature_lyx(pixel_values_recons)
            selected = (input_ids == self.img_context_token_id)
            self.language_model.model.img_idx = torch.where(selected==True)


            thumbnail_cls_attention = thumbnail_cls_attention - thumbnail_cls_attention_recons
            tile_cls_attention = torch.flatten(tile_cls_attention)
            tile_cls_attention_recons = torch.flatten(tile_cls_attention_recons)
            tile_cls_attention = tile_cls_attention - tile_cls_attention_recons

            topk_tile_indices = (tile_cls_attention.topk(32).indices).int().sort().values


            topk_thumbnail_indices = (thumbnail_cls_attention.topk(16).indices+tile_cls_attention.shape[0]).int()

            topk_indices = torch.cat([topk_tile_indices, topk_thumbnail_indices], dim=0).sort().values


            topk_indices = torch.unique(torch.cat([topk_indices, indices_diffusion.to(topk_indices.device)], dim=0)).sort().values


            if generate_kwargs.get('verbose', False):
                print("final topk_indices:", topk_indices.shape)


            image_pad_token_id = 151655
            image_pad_embedding = self.language_model.get_input_embeddings()(torch.tensor([image_pad_token_id], device=vit_embeds.device))

            vit_embeds_flat = vit_embeds.clone().view(-1, vit_embeds.shape[-1])
            vit_embeds_flat[topk_indices] = image_pad_embedding.squeeze(0)
            vit_embeds_pad = vit_embeds_flat.view(vit_embeds.shape[0], vit_embeds.shape[1], vit_embeds.shape[2])


            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)


            input_embeds_pad = input_embeds.clone()
            input_embeds_recons = input_embeds.clone()

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            input_embeds_pad[selected] = vit_embeds_pad.reshape(-1, C).to(input_embeds.device)
            input_embeds_recons[selected] = vit_embeds_useless.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
            input_embeds_pad = input_embeds_pad.reshape(B, N, C)
            input_embeds_recons = input_embeds_recons.reshape(B, N, C)


            input_embeds_textonly = self.language_model.get_input_embeddings()(input_ids_textonly)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)


        generated_ids = input_ids.clone()
        past_key_values = None
        eos_token_id = generate_kwargs.get('eos_token_id', getattr(self.config, 'eos_token_id', 151645))
        if eos_token_id is None:
            eos_token_id = 151645
        max_new_tokens = generate_kwargs.get('max_new_tokens', 1024)
        verbose = generate_kwargs.get('verbose', False)
        next_token_ids = torch.tensor([], dtype=torch.long, device=input_ids.device)

        for step in range(max_new_tokens):

            if step == 0:
                outputs_original = self.language_model(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    use_cache=True,
                )
            else:
                outputs_original = self.language_model(
                    input_ids=next_token_id,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )


            if step == 0:
                outputs_pad_model = self.language_model(
                    inputs_embeds=input_embeds_pad,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
            else:
                outputs_pad_model = self.language_model(
                    input_ids=next_token_id,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values_pad,
                    use_cache=True,
                )

            if step == 0:
                outputs_recons_model = self.language_model(
                    inputs_embeds=input_embeds_recons,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
            else:
                outputs_recons_model = self.language_model(
                    input_ids=next_token_id,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values_recons,
                    use_cache=True,
                )

            if step == 0:
                outputs_textonly_model = self.language_model(
                    inputs_embeds=input_embeds_textonly,
                    attention_mask=attention_mask_textonly,
                    use_cache=True,
                )
            else:
                outputs_textonly_model = self.language_model(
                    input_ids=next_token_id,
                    attention_mask=attention_mask_textonly,
                    past_key_values=past_key_values_textonly,
                    use_cache=True,
                )

            original_probs = torch.softmax(outputs_original.logits[:, -1, :], dim=-1)
            pad_probs = torch.softmax(outputs_pad_model.logits[:, -1, :], dim=-1)
            recons_probs = torch.softmax(outputs_recons_model.logits[:, -1, :], dim=-1)
            textonly_probs = torch.softmax(outputs_textonly_model.logits[:, -1, :], dim=-1)

            top_n = 3


            topn_probs, topn_indices = torch.topk(original_probs[0], top_n, dim=-1)
            topn_probs_pad, topn_indices_pad = torch.topk(pad_probs[0], top_n, dim=-1)
            topn_probs_recons, topn_indices_recons = torch.topk(recons_probs[0], top_n, dim=-1)
            topn_probs_textonly, topn_indices_textonly = torch.topk(textonly_probs[0], top_n, dim=-1)

            if verbose and step == 0:
                print("Top n token probabilities:")
                for i in range(top_n):
                    print(f"  Rank {i+1}: Token ID {topn_indices[i].item()}: {topn_probs[i].item():.15f}")
                print("Top n token probabilities (Pad):")
                for i in range(top_n):
                    print(f"  Rank {i+1}: Token ID {topn_indices_pad[i].item()}: {topn_probs_pad[i].item():.15f}")
                print("Top n token probabilities (Recons):")
                for i in range(top_n):
                    print(f"  Rank {i+1}: Token ID {topn_indices_recons[i].item()}: {topn_probs_recons[i].item():.15f}")
                print("Top n token probabilities (Textonly):")
                for i in range(top_n):
                    print(f"  Rank {i+1}: Token ID {topn_indices_textonly[i].item()}: {topn_probs_textonly[i].item():.15f}")


            def auto_alpha(A_logits, topk=50, a_min=0.1, a_max=0.999, gamma=1.0, temp=1.0):

                probs = torch.softmax(A_logits / temp, dim=-1)

                if topk is not None and topk < probs.size(-1):
                    p, _ = probs.topk(topk, dim=-1)
                    p = p / p.sum(dim=-1, keepdim=True)
                    K = p.size(-1)
                else:
                    p = probs
                    K = p.size(-1)

                HHI = (p * p).sum(dim=-1)
                s = (HHI - 1.0 / K) / (1.0 - 1.0 / K)
                s = s.clamp(0, 1)

                a = a_min + (a_max - a_min) * (s ** gamma)
                return a


            def compute_js_divergence(p, q):

                m = 0.5 * (p + q)


                kl_p_m = torch.sum(p * torch.log(p / (m + 1e-10)), dim=-1)


                kl_q_m = torch.sum(q * torch.log(q / (m + 1e-10)), dim=-1)


                js_divergence = 0.5 * kl_p_m + 0.5 * kl_q_m

                return js_divergence


            contrastive_logits =  1*original_probs - 0.5*pad_probs - 0*recons_probs


            mask = torch.full_like(contrastive_logits[0], float('-inf'))
            mask[topn_indices] = 0


            contrastive_logits = contrastive_logits + mask.unsqueeze(0)


            max_idx = topn_indices[0]
            max_val = contrastive_logits[0, max_idx]


            if max_val < 0:

                for i in range(1, top_n):
                    idx = topn_indices[i]
                    val = contrastive_logits[0, idx]
                    if val >= max_val:
                        if verbose:
                            print("contrastive replacement:", contrastive_logits[0, idx])
                        contrastive_logits[0, idx] = 1
                        break
            else:

                for i in range(1, top_n):
                    idx = topn_indices[i]
                    val = contrastive_logits[0, idx]
                    if val >= max_val:
                        if verbose:
                            print(f"contrastive replacement: {contrastive_logits[0, idx].item():.15f}")
                        contrastive_logits[0, idx] = 1
                        break


            if verbose and step < 2:
                probs = contrastive_logits
                topn_probs, topn_ids = torch.topk(probs, top_n, dim=-1)
                print("Top n token probabilities after contrastive:")
                for i in range(top_n):
                    print(f"Token ID {topn_ids[0, i].item()}: {topn_probs[0, i].item():.15f}")
            next_token_id = torch.argmax(contrastive_logits, dim=-1, keepdim=True)
            next_token_id_original = torch.argmax(outputs_original.logits[:, -1, :], dim=-1, keepdim=True)
            if next_token_id_original.item() == eos_token_id:
                break
            next_token_ids = torch.cat([next_token_ids, next_token_id[0]], dim=-1)


            generated_ids = torch.cat([generated_ids, next_token_id[0]], dim=-1)


            past_key_values = outputs_original.past_key_values
            past_key_values_pad = outputs_pad_model.past_key_values
            past_key_values_recons = outputs_recons_model.past_key_values
            past_key_values_textonly = outputs_textonly_model.past_key_values
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)], dim=-1)
            attention_mask_textonly = torch.cat([attention_mask_textonly, torch.ones((attention_mask_textonly.shape[0], 1), device=attention_mask_textonly.device)], dim=-1)


            if next_token_id.item() == eos_token_id or next_token_id_original.item() == eos_token_id:
                break

        outputs = next_token_ids.unsqueeze(0)

        return outputs


    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, value):
        return self.language_model.set_output_embeddings(value)
