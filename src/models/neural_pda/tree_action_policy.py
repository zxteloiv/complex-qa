from typing import Literal, Tuple
import torch
import torch.nn.functional
from .partial_tree_encoder import TopDownTreeEncoder
from ..interfaces.composer import TwoVecComposer
from ..base_s2s.stacked_encoder import StackedEncoder
from utils.nn import get_final_encoder_states
import utils.nn as utilnn
import allennlp.nn.util as utilsallen


class TreeActionPolicy(torch.nn.Module):
    def __init__(self,
                 node_embedding: torch.nn.Module,
                 pos_embedding: torch.nn.Module,
                 topo_pos_encoder: StackedEncoder,
                 feature_composer: TwoVecComposer,
                 tree_encoder: TopDownTreeEncoder,
                 node_action_mapper: torch.nn.Module,
                 padding: int = 0,
                 ):
        super().__init__()
        self.node_embedding = node_embedding
        self.topo_pos_embedding = pos_embedding
        self.topo_encoder = topo_pos_encoder
        self.topo_encoder.output_every_layer = False

        self.feature_composer = feature_composer
        self.tree_encoder = tree_encoder
        self.action_mapping = node_action_mapper
        self.padding = padding

    def forward(self, tree_nodes: torch.Tensor, node_pos: torch.Tensor, node_parents: torch.Tensor):
        """
        :param tree_nodes: (batch, max_node_num), indices to node symbols
        :param node_pos: (batch, max_node_num, max_tree_depth), the node positional indices along the route to each node
        :param node_parents: (batch, max_node_num), indices to the tree_nodes of the parents
        :return:
        """
        # node_feature: (B, N, node_hid_dim)
        node_feature = self.node_embedding(tree_nodes)
        batch_sz, node_num = node_feature.size()[:2]

        # pos_emb: (B * N, depth, pos_emb)
        pos_emb = self.topo_pos_embedding(node_pos.reshape(batch_sz * node_num, -1))
        pos_mask = (node_pos != self.padding).reshape(batch_sz * node_num, -1)
        # pos_hid: (B * N, depth, pos_hid_dim)
        pos_hid = self.topo_encoder(pos_emb, pos_mask)

        # pos_feature: (B, N, np_dim)
        pos_feature = get_final_encoder_states(pos_hid, pos_mask).reshape(batch_sz, node_num, -1)

        # node_feats: (B, N, feats_dim)
        node_feats = self.feature_composer(pos_feature, node_feature)

        # node_hid: (B, N, hid)
        node_hid, _ = self.tree_encoder(node_feats, node_parents, (tree_nodes != self.padding).long())

        # node_logits: (B, N, action_types)
        node_logits = self.action_mapping(node_hid)

        output = {
            "logits": node_logits,
            "node_mask": (tree_nodes != self.padding)
        }

        return output

    def get_logprob(self, node_logits: torch.Tensor, node_mask: torch.ByteTensor):
        """
        :param node_logits: (B, N, A)
        :param node_mask: (B, N)
        :return:
        """
        # (B, N, 1)
        node_log_mask = (node_mask.unsqueeze(-1) + utilsallen.tiny_value_of_dtype(node_logits.dtype)).log()
        # (B, N, A)
        masked_logits = node_logits + node_log_mask

        # (B, N * A)
        masked_logits_rs = masked_logits.reshape(node_logits.size()[0], -1)
        logprob_rs = utilnn.logits_to_prob(masked_logits_rs, 'bounded')
        # (B, N, A)
        logprob = logprob_rs.reshape(*node_logits.size())
        prob = utilnn.logits_to_prob(masked_logits_rs, 'none').reshape(*node_logits.size())
        return logprob, prob

    def decode(self, prob: torch.Tensor, logprob: torch.Tensor, sample_num: int = 5,
               method: Literal["max", "topk", "multinomial"] = "max"
               ) -> Tuple[list, list, torch.Tensor, torch.Tensor]:
        """
        :param prob: (batch, node_num, action_num)
        :param logprob: (batch, node_num, action_num)
        :param sample_num: int, used when method is not "max"
        :param method: decoding method
        :return: returns the decoded node idx, action idx, and the logprobs and probs, all of size (B, S)
        """
        batch, node_num, action_num = prob.size()

        flat_prob = prob.reshape(batch, -1)

        # (B, S)
        selected_prob = None
        if method == "max":
            selected_prob, indices = torch.max(flat_prob, dim=-1)
        elif method == "topk":
            selected_prob, indices = torch.topk(flat_prob, sample_num, largest=True, dim=-1, sorted=True)
        elif method == 'multinomial':
            indices = torch.multinomial(flat_prob, sample_num)
        else:
            raise ValueError(f'unknown decoding method "{method}" is set')

        node_idx = (indices // action_num)
        action_idx = (indices % action_num)

        # (B, 1)
        batch_dim_indices = torch.arange(batch, device=prob.device).unsqueeze(-1)

        # (B, S)
        selected_logp = logprob[batch_dim_indices, node_idx, action_idx]
        if selected_prob is None:
            selected_prob = prob[batch_dim_indices, node_idx, action_idx]

        return node_idx.tolist(), action_idx.tolist(), selected_logp, selected_prob

