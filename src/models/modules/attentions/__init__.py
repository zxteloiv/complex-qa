import torch

from models.modules.attentions import CatComposer, CatMappingComposer, AddComposer, MappingAddComposer, \
    AdaptiveGeneralAttention, AdaptiveAllenLogits
from models.modules.attentions.adaptive_attention import AdaptiveGeneralAttention, AdaptiveAllenLogits
from models.modules.attentions.attention_composer import (
    CatComposer, AddComposer, MappingAddComposer,
    CatMappingComposer, NoneComposer
)

cls_mappings = {
    "cat": CatComposer,
    "add": AddComposer,
    "mapping_add": MappingAddComposer,
    "cat_mapping": CatMappingComposer,
}


def get_attn_composer(cls_type: str, context_dim: int, vector_dim: int, output_dim: int, activation: str):
    if cls_type in ("none", "passthrough", "linear"):
        return NoneComposer(vector_dim, output_dim, activation)
    elif cls_type == "cat":
        assert output_dim == context_dim + vector_dim, "CatComposer will concatenate the context and vector"
        return CatComposer(context_dim, vector_dim)
    elif cls_type == "cat_mapping":
        return CatMappingComposer(context_dim, vector_dim, output_dim, activation)
    elif cls_type == "add":
        assert context_dim == vector_dim == output_dim, "AddComposer requires all dimensions equal to sum"
        return AddComposer(context_dim)
    elif cls_type == "mapping_add":
        return MappingAddComposer(context_dim, vector_dim, output_dim, activation)
    else:
        raise ValueError(f"The {cls_type} composer is not supported. Current composers {cls_mappings.keys()}")


def get_attention(attn_type: str,
                  vector_dim: int = 0,
                  matrix_dim: int = 0,
                  **kwargs):
    """
    Build an Attention module with specified parameters.
    :param attn_type: indicates the attention type, e.g. "bilinear", "dot_product" or "none"
    :param vector_dim: the vector to compute attention
    :param matrix_dim: the bunch of vectors to be attended against (batch, num, matrix_dim)
    :return: a torch.nn.Module
    """

    attn_type = attn_type.lower()
    if attn_type in ("bilinear", "dot_product"):
        attn_type = 'adaptive_' + attn_type

    if attn_type == 'adaptive_dot_product':
        from .adaptive_logits import DotProductLogits
        from allennlp.modules.matrix_attention import DotProductMatrixAttention
        attn = AdaptiveGeneralAttention(DotProductLogits(vector_dim, matrix_dim))

    elif attn_type == 'adaptive_bilinear':
        from allennlp.modules.matrix_attention import BilinearMatrixAttention
        attn = AdaptiveGeneralAttention(AdaptiveAllenLogits(BilinearMatrixAttention(vector_dim, matrix_dim)))

    elif attn_type == 'adaptive_mha':
        from allennlp.modules.matrix_attention import DotProductMatrixAttention
        num_heads = kwargs.get('num_heads', 8)
        attn = AdaptiveGeneralAttention(
            AdaptiveAllenLogits(DotProductMatrixAttention()),
            init_tau=(vector_dim // num_heads) ** 0.5,
            num_heads=num_heads,
            pre_q_mapping=torch.nn.Linear(vector_dim, matrix_dim),
            pre_k_mapping=torch.nn.Linear(matrix_dim, matrix_dim),
            pre_v_mapping=torch.nn.Linear(matrix_dim, matrix_dim),
            post_ctx_mapping=torch.nn.Linear(matrix_dim, matrix_dim),
        )

    elif attn_type == "none":
        attn = None

    else:
        raise NotImplementedError

    return attn