import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import _canonical_mask, _none_or_dtype, _in_projection_packed, _in_projection, _mha_shape_check #, _check_key_padding_mask
import math
# from torch.nn.activation import _check_arg_device
# from torch.nn.activation import _arg_requires_grad
# from torch.nn.activation import _is_make_fx_tracing

def wrap_multihead_attention(model):
    for i, block in enumerate(model.visual.transformer.resblocks):
        model.visual.transformer.resblocks[i].attn = ScaledMultiheadAttention(block.attn)
    return model

class ScaledMultiheadAttention(nn.Module):
    def __init__(self, original_multihead_attention):
        super().__init__()
        self.original = original_multihead_attention
        self.num_heads = original_multihead_attention.num_heads
        self.head_dim = original_multihead_attention.head_dim
        self.embed_dim = original_multihead_attention.embed_dim
        self.kdim = original_multihead_attention.kdim
        self.vdim = original_multihead_attention.vdim
        self._qkv_same_embed_dim = original_multihead_attention._qkv_same_embed_dim
        self.batch_first = original_multihead_attention.batch_first
        self.dropout = original_multihead_attention.dropout
        self.add_zero_attn = original_multihead_attention.add_zero_attn
        self.bias_k = original_multihead_attention.bias_k
        self.bias_v = original_multihead_attention.bias_v
        self.learned_scale = nn.Parameter(torch.ones(self.num_heads))

    def _reset_parameters(self):
        # Initialize the learned scale parameter
        nn.init.ones_(self.learned_scale)
        # Initialize the original multihead attention parameters
        self.original.reset_parameters()

    def __setstate__(self, state):
        # if "learned_scale" not in state:
        #     # If the learned_scale parameter is not in the state, add it
        #     state["learned_scale"] = torch.ones(self.num_heads)
            # self.original.__setstate__(state)
        # else:
        #     # If the learned_scale parameter is in the state, set it
        #     self.learned_scale = state["learned_scale"]
        #     # Call the original __setstate__ method
        #     del state["learned_scale"]
        #     state = { k.replace("original.", ""): v for k, v in state.items() }
        self.original.__setstate__(state)

    def scaled_multi_head_attention_forward(self, query ,
        key ,
        value ,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight,
        in_proj_bias,
        bias_k,
        bias_v,
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight,
        out_proj_bias,
        training: bool = True,
        key_padding_mask = None,
        need_weights: bool = True,
        attn_mask = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight = None,
        k_proj_weight = None,
        v_proj_weight = None,
        static_k = None,
        static_v = None,
        average_attn_weights: bool = True,
        is_causal: bool = False):
        tens_ops = (
            query,
            key,
            value,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            out_proj_weight,
            out_proj_bias,
        )
        # if has_torch_function(tens_ops):
        #     return handle_torch_function(
        #         multi_head_attention_forward,
        #         tens_ops,
        #         query,
        #         key,
        #         value,
        #         embed_dim_to_check,
        #         num_heads,
        #         in_proj_weight,
        #         in_proj_bias,
        #         bias_k,
        #         bias_v,
        #         add_zero_attn,
        #         dropout_p,
        #         out_proj_weight,
        #         out_proj_bias,
        #         training=training,
        #         key_padding_mask=key_padding_mask,
        #         need_weights=need_weights,
        #         attn_mask=attn_mask,
        #         is_causal=is_causal,
        #         use_separate_proj_weight=use_separate_proj_weight,
        #         q_proj_weight=q_proj_weight,
        #         k_proj_weight=k_proj_weight,
        #         v_proj_weight=v_proj_weight,
        #         static_k=static_k,
        #         static_v=static_v,
        #         average_attn_weights=average_attn_weights,
        #     )

        is_batched = _mha_shape_check(
            query, key, value, key_padding_mask, attn_mask, num_heads
        )

        # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
        # is batched, run the computation and before returning squeeze the
        # batch dimension so that the output doesn't carry this temporary batch dimension.
        if not is_batched:
            # unsqueeze if the input is unbatched
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)

        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        key_padding_mask = _canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=_none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        if is_causal and attn_mask is None:
            raise RuntimeError(
                "Need attn_mask if specifying the is_causal hint. "
                "You may use the Transformer module method "
                "`generate_square_subsequent_mask` to create this mask."
            )

        if is_causal and key_padding_mask is None and not need_weights:
            # when we have a kpm or need weights, we need attn_mask
            # Otherwise, we use the is_causal hint go as is_causal
            # indicator to SDPA.
            attn_mask = None
        else:
            attn_mask = _canonical_mask(
                mask=attn_mask,
                mask_name="attn_mask",
                other_type=None,
                other_name="",
                target_type=query.dtype,
                check_other=False,
            )

            if key_padding_mask is not None:
                # We have the attn_mask, and use that to merge kpm into it.
                # Turn off use of is_causal hint, as the merged mask is no
                # longer causal.
                is_causal = False

        assert (
            embed_dim == embed_dim_to_check
        ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
        else:
            head_dim = embed_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        if use_separate_proj_weight:
            # allow MHA to have different embedding dimensions when separate projection weights are used
            assert (
                key.shape[:2] == value.shape[:2]
            ), f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
        else:
            assert (
                key.shape == value.shape
            ), f"key shape {key.shape} does not match value shape {value.shape}"

        #
        # compute in-projection
        #
        if not use_separate_proj_weight:
            assert (
                in_proj_weight is not None
            ), "use_separate_proj_weight is False but in_proj_weight is None"
            q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        else:
            assert (
                q_proj_weight is not None
            ), "use_separate_proj_weight is True but q_proj_weight is None"
            assert (
                k_proj_weight is not None
            ), "use_separate_proj_weight is True but k_proj_weight is None"
            assert (
                v_proj_weight is not None
            ), "use_separate_proj_weight is True but v_proj_weight is None"
            if in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = in_proj_bias.chunk(3)
            q, k, v = _in_projection(
                query,
                key,
                value,
                q_proj_weight,
                k_proj_weight,
                v_proj_weight,
                b_q,
                b_k,
                b_v,
            )

        # prep attention mask

        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                    )
            else:
                raise RuntimeError(
                    f"attn_mask's dimension {attn_mask.dim()} is not supported"
                )

        # add bias along batch dimension (currently second)
        if bias_k is not None and bias_v is not None:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert bias_k is None
            assert bias_v is None

        #
        # reshape q, k, v for multihead attention and make them batch first
        #
        q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        if static_k is None:
            k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert (
                static_k.size(0) == bsz * num_heads
            ), f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
            assert (
                static_k.size(2) == head_dim
            ), f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
            k = static_k
        if static_v is None:
            v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert (
                static_v.size(0) == bsz * num_heads
            ), f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
            assert (
                static_v.size(2) == head_dim
            ), f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
            v = static_v

        # add zero attention along batch dimension (now first)
        if add_zero_attn:
            zero_attn_shape = (bsz * num_heads, 1, head_dim)
            k = torch.cat(
                [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1
            )
            v = torch.cat(
                [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1
            )
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        # update source sequence length after adjustments
        src_len = k.size(1)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            if not torch.jit.is_scripting() and not torch.jit.is_tracing():
                # _check_key_padding_mask(key_padding_mask, src_len, bsz)
                pass

            key_padding_mask = (
                key_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, num_heads, -1, -1)
                .reshape(bsz * num_heads, 1, src_len)
            )
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        # adjust dropout probability
        if not training:
            dropout_p = 0.0

        #
        # (deep breath) calculate attention and out projection
        #

        if need_weights:
            _B, _Nt, E = q.shape
            q_scaled = q * math.sqrt(1.0 / float(E))

            assert not (
                is_causal and attn_mask is None
            ), "FIXME: is_causal not implemented for need_weights"

            if attn_mask is not None:
                attn_output_weights = torch.baddbmm(
                    attn_mask, q_scaled, k.transpose(-2, -1)
                )
            else:
                attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
            attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            if dropout_p > 0.0:
                attn_output_weights = F.dropout(attn_output_weights, p=dropout_p)

            attn_output = torch.bmm(attn_output_weights, v)

            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            )
            attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            # optionally average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)

            if not is_batched:
                # squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)
                attn_output_weights = attn_output_weights.squeeze(0)
            return attn_output, attn_output_weights
        else:
            # attn_mask can be either (L,S) or (N*num_heads, L, S)
            # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
            # in order to match the input for SDPA of (N, num_heads, L, S)
            if attn_mask is not None:
                if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(0)
                else:
                    attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

            q = q.view(bsz, num_heads, tgt_len, head_dim)
            k = k.view(bsz, num_heads, src_len, head_dim)
            v = v.view(bsz, num_heads, src_len, head_dim)

            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask, dropout_p, is_causal
            )

            # per head scaling
            attn_output = attn_output * self.learned_scale.view(1, -1, 1, 1)

            attn_output = (
                attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
            )

            attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
            if not is_batched:
                # squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)
            return attn_output, None

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
        why_not_fast_path = ""
        if (
            (attn_mask is not None and torch.is_floating_point(attn_mask))
            or (key_padding_mask is not None)
            and torch.is_floating_point(key_padding_mask)
        ):
            why_not_fast_path = "floating-point masks are not supported for fast path."

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        if not is_fastpath_enabled:
            why_not_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        elif not is_batched:
            why_not_fast_path = (
                f"input not batched; expected query.dim() of 3 but got {query.dim()}"
            )
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.original.in_proj_bias is not None and query.dtype != self.original.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.original.in_proj_bias.dtype}) don't match"
        elif self.original.in_proj_weight is None:
            why_not_fast_path = "in_proj_weight was None"
        elif query.dtype != self.original.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.original.in_proj_weight.dtype}) don't match"
        elif self.original.training:
            why_not_fast_path = "training is enabled"
        elif (self.num_heads % 2) != 0:
            why_not_fast_path = "self.num_heads is not even"
        elif not self.original.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.original.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.original.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.original.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self.original._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (
            key_padding_mask is not None or attn_mask is not None
        ):
            why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.original.in_proj_weight,
                self.original.in_proj_bias,
                self.original.out_proj.weight,
                self.original.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            # if torch.overrides.has_torch_function(tensor_args):
            #     why_not_fast_path = "some Tensor argument has_torch_function"
            # elif _is_make_fx_tracing():
            #     why_not_fast_path = "we are running make_fx tracing"
            # elif not all(_check_arg_device(x) for x in tensor_args):
            #     why_not_fast_path = (
            #         "some Tensor argument's device is neither one of "
            #         f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}"
            #     )
            # elif torch.is_grad_enabled() and any(
            #     _arg_requires_grad(x) for x in tensor_args
            # ):
            #     why_not_fast_path = (
            #         "grad is enabled and at least one of query or the "
            #         "input/output projection weights or biases requires_grad"
            #     )
            if not why_not_fast_path:
                merged_mask, mask_type = self.original.merge_masks(
                    attn_mask, key_padding_mask, query
                )

                if self.original.in_proj_bias is not None and self.original.in_proj_weight is not None:
                    return torch._native_multi_head_attention(
                        query,
                        key,
                        value,
                        self.original.embed_dim,
                        self.original.num_heads,
                        self.original.in_proj_weight,
                        self.original.in_proj_bias,
                        self.original.out_proj.weight,
                        self.original.out_proj.bias,
                        merged_mask,
                        need_weights,
                        average_attn_weights,
                        mask_type,
                    )

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, (
            "MultiheadAttention does not support NestedTensor outside of its fast path. "
            + f"The fast path was not hit because {why_not_fast_path}"
        )

        if self.original.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        if not self.original._qkv_same_embed_dim:
            # attn_output, attn_output_weights = F.multi_head_attention_forward(
            attn_output, attn_output_weights = self.scaled_multi_head_attention_forward(
                query,
                key,
                value,
                self.original.embed_dim,
                self.original.num_heads,
                self.original.in_proj_weight,
                self.original.in_proj_bias,
                self.original.bias_k,
                self.original.bias_v,
                self.original.add_zero_attn,
                self.original.dropout,
                self.original.out_proj.weight,
                self.original.out_proj.bias,
                training=self.original.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.original.original.q_proj_weight,
                k_proj_weight=self.original.original.k_proj_weight,
                v_proj_weight=self.original.original.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        else:
            attn_output, attn_output_weights = self.scaled_multi_head_attention_forward(
                query,
                key,
                value,
                self.original.embed_dim,
                self.original.num_heads,
                self.original.in_proj_weight,
                self.original.in_proj_bias,
                self.original.bias_k,
                self.original.bias_v,
                self.original.add_zero_attn,
                self.original.dropout,
                self.original.out_proj.weight,
                self.original.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        if self.original.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

class ScaledAttentionHead(nn.Module):
    '''
    The goal is to scale the output of the attention head by learned coefficient.

    model.visual.transformer.resblock[0] = ScaledAttentionHead(model.visual.transformer.resblock[0])
    '''
    def __init__(self, original_resblock_attn):
        super().__init__()
        self.original = original_resblock_attn
        self.num_heads = original_resblock_attn.num_heads
        # self.head_dim = original_resblock_attn.embed_dim // self.num_heads
        self.head_dim = original_resblock_attn.head_dim
        self.scale = self.head_dim ** -0.5
        self.learned_scale = nn.Parameter(torch.ones(self.num_heads))

    def forward(self, x, attn_mask=None): # need_weights=False, 
        # batch_size, tgt_len, embed_dim = x.shape
        if self.original.batch_first:
            x = x.transpose(0, 1)

        L, N, C = x.shape
        qkv = F.linear(x, self.original.in_proj_weight, bias=self.original.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape(x):
            x = x.reshape(L, N * self.num_heads, -1).transpose(0, 1)
            return x
        
        q = reshape(q)
        k = reshape(k)
        v = reshape(v)

        if self.original.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.original.logit_scale, max=self.original.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.original.attn_drop(attn)
            x = torch.bmm(attn, v)
        else:
            if self.original.use_fsdpa:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.original.attn_drop.p if self.original.training else 0.,
                )
            else:
                q = q * self.scale
                attn = torch.bmm(q, k.transpose(-1, -2))
                if attn_mask is not None:
                    attn += attn_mask
                attn = attn.softmax(dim=-1)
                attn = self.original.attn_drop(attn)
                x = torch.bmm(attn, v)

        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)

        x = x.transpose(0, 1).reshape(L, N, C)

        if self.batch_first:
            x = x.transpose(0, 1)

        x = self.out_proj(x)
        x = self.out_drop(x)
        return x

        # attn_weights = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5
        # if attn_mask is not None and attn_mask.dtype == torch.bool:
        #     # attn_weights += attn_mask
        #     new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        #     new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        #     attn_mask = new_attn_mask

        # if self.original.logit_scale is not None:
            
        # attn_weights = attn_weights.softmax(dim=-1)
        # attn_output = attn_weights @ v

        # attn_output = attn_output * self.scale.view(1, -1, 1, 1)

        # attn_output = attn_output.transpose(1, 2).reshape(batch_size, tgt_len, embed_dim)
        # attn_output = F.linear(attn_output, self.original.attn.out_proj.weight, bias=self.original.attn.out_proj.bias)
        # # attn_output = F.linear(attn_output, self.original.ln_2.weight, bias=self.original.ln_2.bias)
        # # attn_output = self.original.ln_2(attn_output)

        # # attn_output = F.layer_norm(attn_output, self.original.ln_2.weight, self.original.ln_2.bias)
        # # attn_output = self.original.mlp.c_fc(attn_output) + self.original.mlp.c_proj(attn_output)

        # # attn_output = self.original.mlp.c_fc(attn_output)
        # # attn_output = self.original.mlp.c_proj(attn_output)
        # return attn_output

# class ScaledAttentionHead(nn.Module):
#     '''
#     The goal is to scale the output of the attention head by learned coefficient.

#     model.visual.transformer.resblock[0] = ScaledAttentionHead(model.visual.transformer.resblock[0])
#     '''
#     def __init__(self, original_resblock):
#         super().__init__()
#         self.original = original_resblock
#         self.num_heads = original_resblock.attn.num_heads
#         self.head_dim = original_resblock.attn.embed_dim // self.num_heads
#         self.scale = nn.Parameter(torch.ones(self.num_heads))

#     def forward(self, x, attn_mask=None):
#         batch_size, tgt_len, embed_dim = x.shape
#         # x = self.original.ln_1(x)
#         # x = F.linear(x, self.original.ln_1.weight, bias=self.original.ln_1.bias)
#         x = self.original.ln_1(x)

#         qkv = F.linear(x, self.original.attn.in_proj_weight, bias=self.original.attn.in_proj_bias)
#         q, k, v = qkv.chunk(3, dim=-1)

#         def reshape(x):
#             x = x.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
#             return x
        
#         q = reshape(q)
#         k = reshape(k)
#         v = reshape(v)

#         attn_weights = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5
#         if attn_mask is not None:
#             attn_weights += attn_mask
#         attn_weights = attn_weights.softmax(dim=-1)
#         attn_output = attn_weights @ v

#         attn_output = attn_output * self.scale.view(1, -1, 1, 1)

#         attn_output = attn_output.transpose(1, 2).reshape(batch_size, tgt_len, embed_dim)
#         attn_output = F.linear(attn_output, self.original.attn.out_proj.weight, bias=self.original.attn.out_proj.bias)
#         # attn_output = F.linear(attn_output, self.original.ln_2.weight, bias=self.original.ln_2.bias)
#         attn_output = self.original.ln_2(attn_output)

#         # attn_output = F.layer_norm(attn_output, self.original.ln_2.weight, self.original.ln_2.bias)
#         # attn_output = self.original.mlp.c_fc(attn_output) + self.original.mlp.c_proj(attn_output)

#         attn_output = self.original.mlp.c_fc(attn_output)
#         attn_output = self.original.mlp.c_proj(attn_output)
#         return attn_output

