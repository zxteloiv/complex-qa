# the source is largely copied from https://github.com/BerlinerA/Eisner-Diff-Surrogate-PyTorch.git
# with the exception that the function is modified to support mini-batch

import torch
import numpy as np
from torch.autograd import Function, gradcheck
from scipy.special import softmax


def eisner_surrogate(arc_scores, mask=None, hard=False):
    ''' Parse using a differentiable surrogate of Eisner's algorithm.

    Parameters
    ----------
    arc_scores : Tensor
        Arc scores
    hard : bool
        True for discrete (valid) non-differentiable dependency trees,
        False for soft differentiable dependency trees (default is False).
    Returns
    -------
    Tensor
        The highest scoring hard/soft dependency tree
    '''
    return EisnerSurrogate.apply(arc_scores, mask, hard)


R_COMP = 0
L_COMP = 1
R_INCOMP = 2
L_INCOMP = 3


class EisnerSurrogate(Function):    # noqa
    @staticmethod
    def forward(ctx, input, mask=None, hard=False):     # noqa

        # arc_raw_w: (batch, n, n)
        arc_raw_w: torch.Tensor = input.detach()
        batch, n, _ = arc_raw_w.size()

        # use the LAST dim for the batch. although it seems odd, this will save a lot efforts
        # by avoiding using [:] slice for all the w, bp, cw, contrib and their grads.
        # because the algorithm is already hard to read, replacing cw[:, 0, i, j, i:j]
        # with cw[0, i, j, slice] will improve readability to a large extent.
        # this convention only applies to the internal structures,
        # input, mask, and output tensors will not be affected.
        w = arc_raw_w.new_zeros((4, n, n, n, batch))  # weights
        bp = torch.zeros_like(w)
        cw = arc_raw_w.new_zeros((4, n, n, batch))    # cumulative weights
        contrib = torch.zeros_like(cw)

        cw[L_INCOMP, 0, 0] -= float('inf')

        # infer the highest scoring dependency tree
        EisnerSurrogate._inside(n, arc_raw_w, cw, bp, w, hard)

        lengths = mask.sum(-1) - 1 if mask is not None else n - 1
        contrib[R_COMP, 0, lengths, slice(batch)] = 1
        EisnerSurrogate._backptr(n, contrib, bp)

        # print('cw surrogate:')
        # print("R_COMP:\n", cw[R_COMP, :, :, 0].numpy(), sep='')
        # print("L_COMP:\n", cw[L_COMP, :, :, 0].numpy(), sep='')
        # print("R_INCOMP:\n", cw[R_INCOMP, :, :, 0].numpy(), sep='')
        # print("L_INCOMP:\n", cw[L_INCOMP, :, :, 0].numpy(), sep='')
        # print('bp surrogate:')
        # print("R_COMP:\n", bp[R_COMP, :, :, :, 0].argmax(-1).numpy(), sep='')
        # print("L_COMP:\n", bp[L_COMP, :, :, :, 0].argmax(-1).numpy(), sep='')
        # print("R_INCOMP:\n", bp[R_INCOMP, :, :, :, 0].argmax(-1).numpy(), sep='')
        # print("L_INCOMP:\n", bp[L_INCOMP, :, :, :, 0].argmax(-1).numpy(), sep='')
        # print('contrib surrogate:')
        # print("R_COMP:\n", contrib[R_COMP, :, :, 0].numpy(), sep='')
        # print("L_COMP:\n", contrib[L_COMP, :, :, 0].numpy(), sep='')
        # print("R_INCOMP:\n", contrib[R_INCOMP, :, :, 0].numpy(), sep='')
        # print("L_INCOMP:\n", contrib[L_INCOMP, :, :, 0].numpy(), sep='')

        dep_tree = torch.zeros_like(arc_raw_w)  # T_{ij}, (batch, n, n)
        for i in range(n):
            for j in range(1, n):
                if i < j:
                    dep_tree[:, i, j] = contrib[R_INCOMP, i, j]
                elif j < i:
                    dep_tree[:, i, j] = contrib[L_INCOMP, j, i]

        # stash information for backward computation
        ctx.contrib = contrib
        ctx.bp = bp
        ctx.w = w

        return dep_tree

    @staticmethod
    def _inside(n, weights, cw, bp, w, hard):
        def _softmax(x):
            if not hard:
                return torch.softmax(x, dim=0)
            else:  # doesn't support backpropagation when hard=True
                return torch.zeros_like(x).scatter_(0, x.argmax(0).unsqueeze(0), 1)

        def _fill_incomplete(i, j):
            ks, ks1 = slice(i, j), slice(i + 1, j + 1)

            # the incomplete spans, (j - i, batch)
            w_update = cw[R_COMP, i, ks] + cw[L_COMP, ks1, j]
            bp_update = _softmax(w_update)                # (j - i, batch)
            cw_update = (bp_update * w_update).sum(0)     # (batch,)

            # left -> right
            cw[R_INCOMP, i, j] += weights[:, i, j] + cw_update
            # left <- right
            cw[L_INCOMP, i, j] += weights[:, j, i] + cw_update

            # save w and bp, (j - i, batch)
            w[R_INCOMP, i, j, ks] = w_update
            bp[R_INCOMP, i, j, ks] = bp_update
            w[L_INCOMP, i, j, ks] = w_update
            bp[L_INCOMP, i, j, ks] = bp_update

        def _fill_complete(i, j):
            ks, ks1 = slice(i, j), slice(i + 1, j + 1)

            # left -> right, complete
            w_r_update = cw[R_INCOMP, i, ks1] + cw[R_COMP, ks1, j]  # (j - i, batch)
            bp_r_update = _softmax(w_r_update)                      # (j - i, batch)
            cw[R_COMP, i, j] = (bp_r_update * w_r_update).sum(0)    # (batch,)

            # left <- right, complete
            w_l_update = cw[L_COMP, i, ks] + cw[L_INCOMP, ks, j]    # (j - i, batch)
            bp_l_update = _softmax(w_l_update)                      # (j - i, batch)
            cw[L_COMP, i, j] = (bp_l_update * w_l_update).sum(0)    # (batch,)

            # save w and bp
            w[R_COMP, i, j, ks1] = w_r_update
            bp[R_COMP, i, j, ks1] = bp_r_update
            w[L_COMP, i, j, ks] = w_l_update
            bp[L_COMP, i, j, ks] = bp_l_update

        for length in range(1, n):
            for pos in range(n - length):
                _fill_incomplete(pos, pos + length)
                _fill_complete(pos, pos + length)

    @staticmethod
    def _backptr(n, ctb, bp):
        # ctb: (4, n, n, batch), contributions for actions
        # bp: (4, n, n, n, batch)
        def _fill_contrib(i, j):
            ks, ks1 = slice(i, j), slice(i + 1, j + 1)

            # (j - i, batch)
            update_term = ctb[R_COMP, i, j].unsqueeze(0) * bp[R_COMP, i, j, ks1]
            ctb[R_INCOMP, i, ks1] += update_term
            ctb[R_COMP, ks1, j] += update_term

            update_term = ctb[L_COMP, i, j].unsqueeze(0) * bp[L_COMP, i, j, ks]
            ctb[L_COMP, i, ks] += update_term
            ctb[L_INCOMP, ks, j] += update_term

            # by default contrib[3], [2] are w.r.t. bp[3], [2] respectively,
            # but apparently bp[3] == bp[2] because they are both cw[0]+cw[1]
            update_term = ctb[R_INCOMP, i, j].unsqueeze(0) * bp[R_INCOMP, i, j, ks]
            update_term += ctb[L_INCOMP, i, j].unsqueeze(0) * bp[L_INCOMP, i, j, ks]
            ctb[R_COMP, i, ks] += update_term
            ctb[L_COMP, ks1, j] += update_term

        for length in range(n - 1, 0, -1):
            for pos in range(n - length):
                _fill_contrib(pos, pos + length)

    @staticmethod
    def backward(ctx, grad_output):     # noqa

        # unpack stashed information
        contrib, bp, w = ctx.contrib, ctx.bp, ctx.w

        # grads w.r.t. T_{ij}, (batch, n, n)
        grad = grad_output.detach()
        batch, n = grad.size()[:2]

        g_w = grad.new_zeros((4, n, n, n, batch))
        g_bp = torch.zeros_like(g_w)
        g_cw = grad.new_zeros((4, n, n, batch))
        g_contrib = torch.zeros_like(g_cw)

        for i in range(n):
            for j in range(1, n):
                if i < j:
                    g_contrib[R_INCOMP, i, j] = grad[:, i, j]
                elif j < i:
                    g_contrib[L_INCOMP, j, i] = grad[:, i, j]

        # gradient computation
        EisnerSurrogate._backward_backptr(n, g_contrib, g_bp, contrib, bp)
        EisnerSurrogate._backward_inside(n, g_cw, g_bp, g_w, bp, w)

        grad_raw_weights = torch.zeros_like(grad)
        for i in range(n):
            for j in range(1, n):
                if i < j:
                    grad_raw_weights[:, i, j] = g_cw[R_INCOMP, i, j]
                elif j < i:
                    grad_raw_weights[:, i, j] = g_cw[L_INCOMP, j, i]

        return grad_raw_weights, None, None

    @staticmethod
    def _backward_backptr(n: int, g_ctb, g_bp, contrib, bp):
        """
        Find derivatives w.r.t. contrib and bp arr.
        :param n: max length
        :param g_ctb: (4, n, n, batch), gradients for contributions
        :param g_bp: (4, n, n, n, batch), gradients for backtrack pointers
        :param contrib: (4, n, n, batch)
        :param bp: (4, n, n, n, batch)
        :return:
        """
        def _fill_backward_backptr(i, j):
            ks, ks1 = slice(i, j), slice(i + 1, j + 1)

            # incomplete contrib grads
            g_ctb_incomp = (g_ctb[R_COMP, i, ks] + g_ctb[L_COMP, ks1, j])  # (j - i, batch)
            g_ctb_update = (g_ctb_incomp * bp[L_INCOMP, i, j, ks]).sum(0)
            g_ctb[L_INCOMP, i, j] += g_ctb_update
            g_ctb[R_INCOMP, i, j] += g_ctb_update

            g_bp[L_INCOMP, i, j, ks] = g_ctb_incomp * contrib[L_INCOMP, i, j].unsqueeze(0)
            g_bp[R_INCOMP, i, j, ks] = g_ctb_incomp * contrib[R_INCOMP, i, j].unsqueeze(0)

            # right to left complete
            g_ctb_left = g_ctb[L_COMP, i, ks] + g_ctb[L_INCOMP, ks, j]  # (j - i, batch)
            g_ctb[L_COMP, i, j] += (g_ctb_left * bp[L_COMP, i, j, ks]).sum(0)
            g_bp[L_COMP, i, j, ks] = g_ctb_left * contrib[L_COMP, i, j].unsqueeze(0)

            # left to right complete
            g_ctb_right = g_ctb[R_INCOMP, i, ks1] + g_ctb[R_COMP, ks1, j]
            g_ctb[R_COMP, i, j] += (g_ctb_right * bp[R_COMP, i, j, ks1]).sum(0)
            g_bp[R_COMP, i, j, ks1] = g_ctb_right * contrib[R_COMP, i, j].unsqueeze(0)

        for l in range(1, n):
            for i in range(n - l):
                _fill_backward_backptr(i, i + l)

    @staticmethod
    def _backward_inside(n, g_cw, g_bp, g_w, bp, w):

        for l in range(n - 1, 0, -1):
            for i in range(0, n - l):
                j = i + l

                ks, ks1 = slice(i, j), slice(i + 1, j + 1)

                # right to left complete
                g_bp[L_COMP, i, j, ks] += g_cw[L_COMP, i, j].unsqueeze(0) * w[L_COMP, i, j, ks]
                g_w[L_COMP, i, j, ks] += g_cw[L_COMP, i, j].unsqueeze(0) * bp[L_COMP, i, j, ks]
                s = (g_bp[L_COMP, i, j, ks] * bp[L_COMP, i, j, ks]).sum(0, keepdim=True)    # (1, batch)
                g_w[L_COMP, i, j, ks] += bp[L_COMP, i, j, ks] * (g_bp[L_COMP, i, j, ks] - s)
                g_cw[L_COMP, i, ks] += g_w[L_COMP, i, j, ks]
                g_cw[L_INCOMP, ks, j] += g_w[L_COMP, i, j, ks]

                # left to right complete
                g_bp[R_COMP, i, j, ks1] += g_cw[R_COMP, i, j].unsqueeze(0) * w[R_COMP, i, j, ks1]
                g_w[R_COMP, i, j, ks1] += g_cw[R_COMP, i, j].unsqueeze(0) * bp[R_COMP, i, j, ks1]
                s = (g_bp[R_COMP, i, j, ks1] * bp[R_COMP, i, j, ks1]).sum(0, keepdim=True)  # (1, batch)
                g_w[R_COMP, i, j, ks1] += bp[R_COMP, i, j, ks1] * (g_bp[R_COMP, i, j, ks1] - s)
                g_cw[R_INCOMP, i, ks1] += g_w[R_COMP, i, j, ks1]
                g_cw[R_COMP, ks1, j] += g_w[R_COMP, i, j, ks1]

                # right to left incomplete
                g_bp[L_INCOMP, i, j, ks] += g_cw[L_INCOMP, i, j].unsqueeze(0) * w[L_INCOMP, i, j, ks]
                g_w[L_INCOMP, i, j, ks] += g_cw[L_INCOMP, i, j].unsqueeze(0) * bp[L_INCOMP, i, j, ks]
                s = (g_bp[L_INCOMP, i, j, ks] * bp[L_INCOMP, i, j, ks]).sum(0, keepdim=True) # (1, batch)
                g_w[L_INCOMP, i, j, ks] += bp[L_INCOMP, i, j, ks] * (g_bp[L_INCOMP, i, j, ks] - s)
                g_cw[R_COMP, i, ks] += g_w[L_INCOMP, i, j, ks]
                g_cw[L_COMP, ks1, j] += g_w[L_INCOMP, i, j, ks]

                # left to right incomplete
                g_bp[R_INCOMP, i, j, ks] += g_cw[R_INCOMP, i, j].unsqueeze(0) * w[R_INCOMP, i, j, ks]
                g_w[R_INCOMP, i, j, ks] += g_cw[R_INCOMP, i, j].unsqueeze(0) * bp[R_INCOMP, i, j, ks]
                s = (g_bp[R_INCOMP, i, j, ks] * bp[R_INCOMP, i, j, ks]).sum(0, keepdim=True) # (1, batch)
                g_w[R_INCOMP, i, j, ks] += bp[R_INCOMP, i, j, ks] * (g_bp[R_INCOMP, i, j, ks] - s)
                g_cw[R_COMP, i, ks] += g_w[R_INCOMP, i, j, ks]
                g_cw[L_COMP, ks1, j] += g_w[R_INCOMP, i, j, ks]


if __name__ == '__main__':

    from eisner import parse_proj

    fails_count = 0
    n_tests = 10
    for idx in range(n_tests):
        dim = np.random.randint(low=4, high=12)
        scores = torch.randn((3, dim, dim), requires_grad=True, dtype=torch.float64)

        # dim = 4
        # scores = torch.zeros((3, dim, dim), dtype=torch.float64)
        # scores[0][0, 2] = 1
        # scores[0][2, 1] = 1
        # scores[0][2, 3] = 1
        #
        # scores[1][0, 1] = 1
        # scores[1][1, 2] = 1
        # scores[1][2, 3] = 1
        #
        # scores[2][0, 3] = 1
        # scores[2][3, 1] = 1
        # scores[2][1, 2] = 1
        # scores = scores.clone().requires_grad_(True)

        print('-----------+-----------+-----------')
        print(f'Test #{idx + 1} (dim = {dim})')
        # test forward pass
        t_ij = eisner_surrogate(scores, None, True)
        torch_out = t_ij.argmax(1)[:, 1:]
        np_out = np.stack([parse_proj(instance.detach().numpy())[1:] for instance in scores])

        f_test_res = np.array_equal(torch_out, np_out)

        # f_test_res = np.array_equal(
        #         torch.argmax(eisner_surrogate(scores, hard=True), dim=0)[1:],
        #         parse_proj(scores.detach().numpy())[1:])
        #
        print('---------' * 5)
        print(f'Forward pass - {"succeeded" if f_test_res else "failed"}')
        if not f_test_res:
            print(t_ij)
            print('torch_out:', torch_out.tolist())
            print('np_out:', np_out.tolist())
        # test backward pass
        b_test_res = gradcheck(eisner_surrogate, (scores, None, False), eps=1e-6, atol=1e-4)
        print(f'Backward pass - {"succeeded" if b_test_res else "failed"}')

        if not f_test_res or not b_test_res:
            fails_count += 1

    print('-----------+-----------+-----------')
    print(f'Summary: {n_tests - fails_count}/{n_tests} successes | {fails_count}/{n_tests} failures')
