#
import torch
from .partial_tree_encoder import TopDownTreeEncoder
from ..interfaces.composer import TwoVecComposer
from ..base_s2s.stacked_encoder import StackedEncoder
from utils.nn import get_final_encoder_states
from allennlp.nn.util import masked_log_softmax, masked_softmax

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

        logprob = self.get_logprob(node_logits, (tree_nodes != self.padding).long())
        return node_logits, logprob

    def get_logprob(self, node_logits, mask):
        while mask.dim() < node_logits.dim():
            mask = mask.unsqueeze(-1)
        logprob = masked_log_softmax(node_logits, mask)
        return logprob

    def decode(self, masked_logprob: torch.Tensor):
        # max_logprob_each_node, action_each_node: (B, N)
        max_logprob_each_node, action_each_node = torch.max(masked_logprob, dim=-1)

        # max_logprob, node_idx: (B,)
        max_logprob, node_idx = torch.max(max_logprob_each_node, dim=-1)

        # action: (B,)
        action = action_each_node[torch.arange(node_idx.size()[0], device=node_idx.device), node_idx]
        return node_idx.tolist(), action.tolist()




