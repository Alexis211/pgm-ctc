from theano import tensor, scan

from blocks.bricks import Brick

# T: INPUT_SEQUENCE_LENGTH
# B: BATCH_SIZE
# L: OUTPUT_SEQUENCE_LENGTH
# C: NUM_CLASSES
class CTC(Brick):
    def apply(l, probs, l_mask=None, probs_mask=None):
        """
        Numeration:
            Characters 0 to C-1 are true characters
            Character C is the blank character
        Inputs:
            l : L x B : the sequence labelling
            probs : T x B x C+1 : the probabilities output by the RNN
            l_mask : L x B
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
        l_blk = tensor.zeros((S, B))
        l_blk = tensor.set_subtensor(l_blk[1::2,:],l)

        # dimension of alpha :
        #   T x B x S
        # dimension of c :
        #   T x B
        # first value of alpha (size B x S)
        alpha0 = tensor.concatenate([
                                        probs[0, :, C],
                                        probs[0][tensor.arange(B), l[0]],
                                        tensor.zeros((B, S-2))
                                    ], axis=1)
        c0 = alpha0.sum(axis=2)

        # recursion
        def recursion(p, p_mask, prev_alpha, prev_c):
            # TODO
            return prev_alpha[-1], prev_c[-1]

        # apply the recursion with scan
        alpha, c = tensor.scan(fn=recursion,
                               sequences=[probs, probs_mask],
                               outputs_info=[alpha0, c0])

        # return the probability of the labellings


    
    def best_path_decoding(y_hat, y_hat_mask=None):
        # Easy one !
        pass

    def prefix_search(y_hat, y_hat_mask=None):
        # Hard one...
        pass
        
        
 
# vim: set sts=4 ts=4 sw=4 sw=4 tw=0 et:
