import numpy

from theano import tensor, scan

from blocks.bricks import Brick

# T: INPUT_SEQUENCE_LENGTH
# B: BATCH_SIZE
# L: OUTPUT_SEQUENCE_LENGTH
# C: NUM_CLASSES
class CTC(Brick):
    def apply(self, l, probs, l_len=None, probs_mask=None):
        """
        Numeration:
            Characters 0 to C-1 are true characters
            Character C is the blank character
        Inputs:
            l : L x B : the sequence labelling
            probs : T x B x C+1 : the probabilities output by the RNN
            l_len : B : the length of each labelling sequence
            probs_mask : T x B
        Output: the B probabilities of the labelling sequences
        Steps:
            - Calculate y' the labelling sequence with blanks
            - Calculate the recurrence relationship for the alphas
            - Calculate the sequence of the alphas
            - Return the probability found at the end of that sequence
        """
        T = probs.shape[0]
        C = probs.shape[2]-1
        L = l.shape[0]
        S = 2*L+1
        B = l.shape[1]
        
        # l_blk = l with interleaved blanks
        l_blk = C * tensor.ones((S, B), dtype='int32')
        l_blk = tensor.set_subtensor(l_blk[1::2,:],l)
        l_blk = l_blk.T     # now l_blk is B x S

        # dimension of alpha (corresponds to alpha hat in the paper) :
        #   T x B x S
        # dimension of c :
        #   T x B
        # first value of alpha (size B x S)
        alpha0 = tensor.concatenate([
                                        probs[0, :, C][:,None],
                                        probs[0][tensor.arange(B), l[0]][:,None],
                                        tensor.zeros((B, S-2))
                                    ], axis=1)
        c0 = alpha0.sum(axis=1)
        alpha0 = alpha0 / c0[:,None]

        # recursion
        l_blk_2 = tensor.concatenate([-tensor.ones((B,2)), l_blk[:,:-2]], axis=1)
        l_case2 = tensor.neq(l_blk, C) * tensor.neq(l_blk, l_blk_2)
        # l_case2 is B x S

        def recursion(p, p_mask, prev_alpha, prev_c):
            # p is B x C+1
            # prev_alpha is B x S 
            prev_alpha_1 = tensor.concatenate([tensor.zeros((B,1)),prev_alpha[:,:-1]], axis=1)
            prev_alpha_2 = tensor.concatenate([tensor.zeros((B,2)),prev_alpha[:,:-2]], axis=1)

            alpha_bar = prev_alpha + prev_alpha_1
            alpha_bar = tensor.switch(l_case2, alpha_bar + prev_alpha_2, alpha_bar)
            next_alpha = alpha_bar * p[tensor.arange(B)[:,None].repeat(S,axis=1).flatten(), l_blk.flatten()].reshape((B,S))
            next_alpha = tensor.switch(p_mask[:,None], next_alpha, prev_alpha)
            next_c = next_alpha.sum(axis=1)
            
            return next_alpha / next_c[:, None], next_c

        # apply the recursion with scan
        [alpha, c], _ = scan(fn=recursion,
                             sequences=[probs, probs_mask],
                             outputs_info=[alpha0, c0])

        # return the log probability of the labellings
        return tensor.log(c).sum(axis=0)

    
    def best_path_decoding(self, probs, probs_mask=None):
        # probs is T x B x C+1
        T = probs.shape[0]
        B = probs.shape[1]
        C = probs.shape[2]-1

        maxprob = probs.argmax(axis=2)

        # returns two values :
        # label : (T x) T x B
        # label_length : (T x) B
        def recursion(maxp, p_mask, label_length, label):
            label_length = label_length[-1]
            label = label[-1]

            nonzero = p_mask * tensor.ne(maxp, C)
            nonzero_id = nonzero.nonzero()[0]

            new_label = tensor.set_subtensor(label[label_length[nonzero_id], nonzero_id], maxp[nonzero_id])
            new_label_length = tensor.switch(nonzero, label_length + numpy.int32(1), label_length)

            return new_label_length, new_label
            
        label_length, label = tensor.scan(fn=recursion,
                                          sequences=[maxprob, probs_mask],
                                          outputs_info=[tensor.zeros((B),dtype='int32'),tensor.zeros((T,B))])

        return label[-1], label_length[-1]

    def prefix_search(self, probs, probs_mask=None):
        # Hard one...
        pass
        
        
 
# vim: set sts=4 ts=4 sw=4 sw=4 tw=0 et:
