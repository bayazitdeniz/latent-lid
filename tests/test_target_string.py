import unittest

import torch

from lenses import UnembedInfo
from target_string import score_teacher_forced_one_candidate


class IdentityDecodingLens:
    def project_layer(self, hidden_states, layer_idx):
        return hidden_states

    def project_all_layers(self, latents, layer_indices):
        return latents


class TargetStringScoringTests(unittest.TestCase):
    def test_teacher_forced_multitoken_scores(self):
        vocab_size = 4
        weight = torch.eye(vocab_size)
        norm_weight = weight / weight.norm(dim=1, keepdim=True)
        info = UnembedInfo(
            weight=weight,
            bias=None,
            normalized_weight=norm_weight,
            avg_uu=torch.tensor(1.0),
            norm_module=None,
        )
        # latents: (layers=1, seq=4, hidden=vocab)
        latents = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0, 4.0],
                ]
            ]
        )
        prompt_len = 2
        target_tokens = torch.tensor([2, 3], dtype=torch.long)
        step_logprobs, vocab_entropy_bits = score_teacher_forced_one_candidate(
            latents=latents,
            prompt_len=prompt_len,
            target_tokens=target_tokens,
            unembed_info=info,
            decoding_lens=IdentityDecodingLens(),
            layer_indices=[0],
        )
        self.assertIsNone(vocab_entropy_bits)
        log_probs = torch.log_softmax(latents[0], dim=-1)
        expected = log_probs[1, 2] + log_probs[2, 3]
        self.assertTrue(torch.allclose(step_logprobs.sum(dim=-1), expected.unsqueeze(0)))

    def test_teacher_forced_out_of_range_raises(self):
        weight = torch.eye(2)
        norm_weight = weight / weight.norm(dim=1, keepdim=True)
        info = UnembedInfo(
            weight=weight,
            bias=None,
            normalized_weight=norm_weight,
            avg_uu=torch.tensor(1.0),
            norm_module=None,
        )
        latents = torch.zeros((1, 2, 2))
        prompt_len = 2
        target_tokens = torch.tensor([1, 0], dtype=torch.long)
        with self.assertRaises(IndexError):
            score_teacher_forced_one_candidate(
                latents=latents,
                prompt_len=prompt_len,
                target_tokens=target_tokens,
                unembed_info=info,
                decoding_lens=IdentityDecodingLens(),
                layer_indices=[0],
            )


if __name__ == "__main__":
    unittest.main()
