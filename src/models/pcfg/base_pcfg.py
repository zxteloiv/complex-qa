from typing import Tuple

import torch


class PCFGModule(torch.nn.Module):
    """
    A PCFG Module is the interface defined for the general-purpose PCFGs.
    Thus any inherited class had better not implement other interfaces defined for other model.

    e.g. an EmbedAndEncode implementation is restricted to be the encoder bundle of the Seq2seq,
    but a PCFG can be more than an encoder.
    therefore a PCFG extends (PCFGModule, EmbedAndEncode) are better avoided.
    """

    def inference(self, x):
        raise NotImplementedError

    def get_pcfg_params(self, z):
        raise NotImplementedError

    def inside(self, x, pcfg_params, x_hid=None):
        raise NotImplementedError

    def get_encoder_output_size(self):
        raise NotImplementedError

    @staticmethod
    def get_inside_coordinates(n: int, width: int, device):
        un_ = torch.unsqueeze
        tr_ = torch.arange
        lvl_b = un_(tr_(width, device=device), 0)  # (pos=1, sublvl)
        pos_b = un_(tr_(n - width, device=device), 1)  # (pos, subpos=1)
        lvl_c = un_(tr_(width - 1, -1, -1, device=device), 0)  # (pos=1, sublvl), reversed lvl_b
        pos_c = un_(tr_(1, width + 1, device=device), 0) + pos_b  # (pos=(n-width), subpos=width))
        return lvl_b, pos_b, lvl_c, pos_c

    @staticmethod
    def inside_chart_select(score_chart, coordinates, detach: bool = False):
        lvl_b, pos_b, lvl_c, pos_c = coordinates
        # *_score: (batch, pos=(n - width), arrangement=width, NT_T)
        if detach:
            b_score = score_chart[:, lvl_b, pos_b].detach()
            c_score = score_chart[:, lvl_c, pos_c].detach()
        else:
            b_score = score_chart[:, lvl_b, pos_b].clone()
            c_score = score_chart[:, lvl_c, pos_c].clone()
        return b_score, c_score

    def set_condition(self, conditions):
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, pcfg_params, max_steps: int):
        from ..modules.batched_stack import TensorBatchStack
        # roots: (b, NT)
        roots = pcfg_params[0]
        batch_sz = roots.size()[0]

        stack = TensorBatchStack(batch_sz, self.NT, 1, dtype=torch.long, device=roots.device)
        output = TensorBatchStack(batch_sz, self.NT, 1, dtype=torch.long, device=roots.device)

        succ = stack.push(roots.argmax(-1, keepdim=True), push_mask=None)
        step: int = 0
        stack_not_empty: torch.BoolTensor = stack.top()[1] > 0    # (batch,)
        while succ.bool().any() and stack_not_empty.any() and step < max_steps:
            # (batch, 1), (batch,)
            token, pop_succ = stack.pop(stack_not_empty * succ)   # only nonterm tokens stored on the stack
            succ *= pop_succ
            lhs_is_nt = (token < self.NT).squeeze()
            lhs_not_nt = ~lhs_is_nt

            words = self.generate_next_term(pcfg_params, token, lhs_not_nt * succ)
            output.push(words, push_mask=lhs_not_nt * succ)

            rhs_b, rhs_c = self.generate_next_nonterms(pcfg_params, token, lhs_is_nt * succ)
            succ *= stack.push(rhs_c, lhs_is_nt * succ)
            succ *= stack.push(rhs_b, lhs_is_nt * succ)

            stack_not_empty: torch.BoolTensor = stack.top()[1] > 0    # (batch,)
            step += 1

        buffer, mask = output.dump()
        return buffer, mask

    def generate_next_term(self, pcfg_params, token, term_mask) -> torch.LongTensor:
        raise NotImplementedError

    def generate_next_nonterms(self, pcfg_params, token, nonterm_mask) -> Tuple[torch.LongTensor, torch.LongTensor]:
        raise NotImplementedError
