import torch.nn.functional as F
import torch 
import os 
import json 
import einops
from utils import BeamHypotheses

class FakeModel():
    def __init__(self) -> None:
        self.vocabs = {'SOS': 0, 'The':1, 'dog': 2, 'nice': 3, 'car': 4, 'woman': 5, 'house': 6, 'guy': 7, 'and': 8, 'runs': 9, 'has': 10, 'is': 11, 'drives': 12, 'turns': 13, 'EOS': 14, 'UNK': 15, 'PAD': 16}
        self.inverted_vocabs = {v: k for k, v in self.vocabs.items()}
        self.vocab_size = len(self.vocabs)
        self.next_word = {'SOS':  {'dog': 0.4, 'nice': 0.5, 'car':0.1}, 'nice': {'woman': 0.4, 'house': 0.3, 'guy': 0.3}, 'car': {'is': 0.3, 'drives': 0.5, 'turns': 0.2}, 'dog': {'and': 0.05, 'runs': 0.05, 'has': 0.9}}


    def __call__(self, x):
        """fake forward pass

        Args:
            x (_type_): x does not matter, basically a dummy input
        """        
        # decoding
        # start with <sos> token
        # torch.Size([1, 1])
        cap_output = torch.tensor(
            [self.vocabs["SOS"]], device=x.device
        ).unsqueeze(0)
        cap_output = einops.repeat(
            cap_output, "b s ->() (l b) (s k)", l=2, k=49
        )  # torch.Size([1, max_num_sent, 50])
        # mask
        cap_masks = torch.zeros(
            (1, cap_output.shape[1], 49),
            device=x.device,
            dtype=torch.float32,
        )  # torch.Size([1, max_num_sent, 50])
        probs_cap = None
        word_cap = None
        cap_masks[..., 0] = 1.0
        for i in range(49):
            cap_output = cap_output.clone()
            if i != 0:
                cap_output[..., i] = next_token
                cap_masks[..., i] = 1.0

            output_sent = self.pred(cap_output, i) # torch.Size([max_num_sent, 50, vocab_size])
            
            next_token = self.greedy(output_sent, i) # torch.Size([max_num_sent, 50])
            if probs_cap is None:
                probs_cap = output_sent[:, i].unsqueeze(1)
            else:
                probs_cap = torch.cat(
                    (probs_cap, output_sent[:, i].unsqueeze(1)), dim=1
                )
            if word_cap is None:
                word_cap = next_token.unsqueeze(1)
            else:
                word_cap = torch.cat((word_cap, next_token.unsqueeze(1)), dim=1)

        return word_cap, probs_cap
    def pred(self, x, i=0):
        """The goal is to return a tensor of size [max_num_sent, 50, vocab_size] with the probability of the next word res[i] is non empty.

        Args:
            x (_type_): _description_
            i (_type_): _description_

        Returns:
            _type_: _description_
        """        
        # torch.Size([1, max_num_sent, 50])
        res = torch.zeros((x.shape[1], x.shape[2], self.vocab_size), device=x.device)
        x = einops.rearrange(x, "b n w -> (b n) w")
        cap  = x[:, i] # torch.Size([max_num_sent])
        for idx, c in enumerate(cap):
            if self.inverted_vocabs[int(c)] in self.next_word:
                for word, prob in self.next_word[self.inverted_vocabs[int(c)]].items():
                    res[idx, i, self.vocabs[word]] = prob
            else: 
                res[idx, i, self.vocabs['EOS']] = 1.0
        return res * 3
    def get_hidden_states(self, x, i):
        return torch.rand((x.shape[1], x.shape[2], 512), device=x.device)

    def greedy(self, x, i):
        next_token = torch.argmax(
            x[:, i], dim=1
        )  # torch.Size([max_num_sent, 50])
        return next_token
    

    def beam_search(self, x_input, i, beam_size = 3):
        """Function for beam search

        Args:
            x (_type_): x does not matter, basically a dummy input
        """        
        # decoding
        # start with <sos> token
        # torch.Size([1, 1])
        bs = 2 # max_num_sent
        max_len = 50
        cap_output = torch.tensor(
            [self.vocabs["SOS"]], device=x_input.device
        ).unsqueeze(0)
        cap_output = einops.repeat(
            cap_output, "b s ->() (l b) (s k)", l=2*beam_size, k=max_len
        )  # torch.Size([1, max_num_sent*beam_size = 3, 50])
        # mask
        cap_masks = torch.zeros(
            (1, cap_output.shape[1], max_len),
            device=x_input.device,
            dtype=torch.float32,
        )  # torch.Size([1, max_num_sent*beam_size = 3, 50])
        
        probs_cap = None
        word_cap = None
        beam_scores = x_input.new(bs, beam_size).fill_(0)
        beam_scores[:, 1:] = -1e9 # to mask the first top k, to pick only top k=1 at the first step
        beam_scores = beam_scores.view(-1)
        cap_masks[..., 0] = 1.0
        done = [False for _ in range(bs)]
        
        length_penalty = 1.0
        generated_hyps = [BeamHypotheses(beam_size, max_len, length_penalty, False) for _ in range(bs)]
        generated = cap_output.new(bs*beam_size, max_len).fill_(self.vocabs["SOS"])
        for cur_len in range(max_len):
            # cap_output = generated.clone()
            if cur_len != 0:
                cap_output = cap_output[:, beam_idx, ...]
                cap_output[..., cur_len] = beam_words
                cap_masks[..., cur_len] = 1.0

            output_sent = self.pred(cap_output, cur_len) # torch.Size([max_num_sent, 50, vocab_size])
            tensor_we_care = output_sent[:, cur_len] # torch.Size([max_num_sent, vocab_size])
            # Set score to zero where EOS has been reached
                                                
            scores = F.log_softmax(tensor_we_care, dim=-1)       # (bs * beam_size, self.vocab_size)
            
            assert scores.size() == (bs * beam_size, self.vocab_size), f"{scores.size()} vs {(bs * beam_size, self.vocab_size)}"

            # select next words with scores
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (bs * beam_size, self.vocab_size)
            _scores = _scores.view(bs, beam_size * self.vocab_size)            # (bs, beam_size * self.vocab_size)

            next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=1, largest=True, sorted=True)
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

            # next batch beam content
            # list of (bs * beam_size) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for sent_id in range(bs):

                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item())
                if done[sent_id]:
                    next_batch_beam.extend([(0, self.vocabs["PAD"], 0)] * beam_size)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = idx // self.vocab_size
                    word_id = idx % self.vocab_size

                    # end of sentence, or next word
                    if word_id == self.vocabs["EOS"] or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(generated[sent_id * beam_size + beam_id, :cur_len].clone(), value.item())
                    else:
                        next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, self.vocabs["PAD"], 0)] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = x_input.new([x[2] for x in next_batch_beam]).long()

            # re-order batch and internal states
            generated = generated[beam_idx, ... ]
            generated[..., cur_len] = beam_words
            # stop when we are done with each sentence
            if all(done):
                break

        tgt_len = x_input.new(bs).long()
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # generate target batch
        decoded = x_input.new(tgt_len.max().item(), bs).fill_(self.vocabs["PAD"])
        for i, hypo in enumerate(best):
            decoded[:tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = self.vocabs["EOS"]

        # sanity check
        # assert (decoded == self.vocabs["EOS"]).sum() >= 2 * bs, f'{(decoded == self.vocabs["EOS"]).sum()} vs {2 * bs}'
        # if i use this beam search, i wont be able to compute the loss of the model
        return decoded, tgt_len
    

    def contrastive_search(self, x_input, beam_size=3):
        """Implementation of contrastive search 

        It is safer to for loop each batch because I am not sure the parallel behavior yet. 

        Args:
            x_input (_type_): _description_
            beam_size (int, optional): _description_. Defaults to 3.
        """     
        max_num_sent = 1 # bs = 1 wont make the code get error at .item() below. I'll keep it this way for now 
        bs = max_num_sent # max_num_sent
        max_len = 50
        cap_output = torch.tensor(
            [self.vocabs["SOS"]], device=x_input.device
        ).unsqueeze(0)
        cap_output = einops.repeat(
            cap_output, "b s ->() (l b) (s k)", l=bs, k=max_len
        )  # torch.Size([1, max_num_sent*beam_size = 3, 50])
        # mask
        cap_masks = torch.zeros(
            (1, cap_output.shape[1], max_len),
            device=x_input.device,
            dtype=torch.float32,
        )  # torch.Size([1, max_num_sent*beam_size = 3, 50])
        generated = []
        for cur_len in range(max_len-1):
            cap_output = cap_output.clone() # clone to avoid duplicate reference from einops.repeat (set value to x will change all tensor into x)
            if cur_len != 0:
                cap_output[..., cur_len] = token
                cap_masks[..., cur_len] = 1.0
            # if cur_len== 0: # first step 
            output = self.pred(cap_output, cur_len) # torch.Size([max_num_sent*beam_size, 50, vocab_size])
            last_hidden_states = self.get_hidden_states(cap_output, cur_len)    # [B, S, E]
            logit_for_next_step = output[:, cur_len, :]    # [B, V]

            bsz, seqlen, embed_dim = last_hidden_states.size()
            next_probs = F.softmax(logit_for_next_step, dim=-1)
            _, top_k_ids = torch.topk(logit_for_next_step, dim=-1, k=beam_size)    # [B, K]
            top_k_probs = torch.gather(next_probs, dim=1, index=top_k_ids)    # [B, K]
            tmp_cap_output = torch.cat([cap_output for _ in range(beam_size)]).clone()
            tmp_cap_output[..., cur_len+1] = top_k_ids.reshape(-1)
            output = self.pred(tmp_cap_output, cur_len+1)
            logits = output[:, cur_len+1, :]    # [B*K, V]
            next_hidden = self.get_hidden_states(cap_output, cur_len+1)
            next_hidden = next_hidden[:,cur_len+1:cur_len+2,:]    # [B*K, 1, E], trick to keep the dim 1 
            context_hidden = last_hidden_states.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(bsz*beam_size, seqlen, embed_dim)    # [B*K, S, E]
            selected_idx = self.ranking_fast(
                context_hidden, 
                next_hidden, 
                top_k_probs,    # [B, K] 
                0.5,
                beam_size,
            )     # [B]
            # prepare for the next step
            next_id = top_k_ids[range(len(top_k_ids)), selected_idx].unsqueeze(-1)    # [B, 1]
            next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), beam_size))    # [B, K, E]
            next_hidden = next_hidden[range(bsz), selected_idx, :]    # [B, E]
            last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)    # [B, S, E]
            logit_for_next_step = torch.stack(torch.split(logits, beam_size))[range(bsz), selected_idx, :]    # [B, V]
            token = next_id.squeeze(dim=-1).item()
            generated.append(token)
        return generated

    def ranking_fast(self, context_hidden, next_hidden, next_top_k_probs, alpha, beam_width):
        '''from https://github.com/yxuansu/SimCTG/blob/main/SimCTGEncDec/SimCTGBART/utlis.py

        Args:
            context_hidden: bsz*beam x seqlen x embed_dim
            next_hidden: bsz*beam x 1 x embed_dim
            next_top_k_probs: bsz x beam
        '''
        _, context_len, embed_dim = context_hidden.size()
        norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
        norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
        cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1,2)).squeeze(-1)    # [B*K, S]
        scores, _ = torch.max(cosine_matrix, dim=-1)    # [B*K]
        next_top_k_probs = next_top_k_probs.view(-1)    # [B*K]
        scores = (1.0 - alpha) * next_top_k_probs - alpha * scores 
        scores = torch.stack(torch.split(scores, beam_width))    # [B, K]
        selected_idx = scores.max(dim=-1)[1]    # [B]
        return selected_idx

            
fk = FakeModel()
# w, p = fk(torch.rand(1, 3, 224, 224).cuda())
# print(w, p)
# decoded, tgt_len = fk.beam_search(torch.rand(1, 3, 224, 224).cuda(), 0)
# print(decoded, tgt_len)
decoded = fk.contrastive_search(torch.rand(1, 3, 224, 224).cuda(), 1)
print(decoded)