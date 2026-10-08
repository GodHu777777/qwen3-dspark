import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from dspark_qwen.cached_target import CachedTarget
from dspark_qwen.packed_target import PackedTarget


def ids(*tokens):
    return torch.tensor([tokens], dtype=torch.long)


class PackedTargetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(191)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=128, attention_dropout=0.0)
        config._attn_implementation = 'sdpa'
        self.model = Qwen3ForCausalLM(config).eval()
        self.target = PackedTarget(self.model, layer_ids=(0, 1))
        self.prefixes, self.oracles = {}, {}

    def add(self, name):
        self.target.add_request(name)
        self.prefixes[name] = ids()
        self.oracles[name] = CachedTarget(self.model, layer_ids=(0, 1))

    def assert_kv(self):
        for name in self.prefixes:
            if not self.prefixes[name].numel():
                continue
            actual = self.target.request_kv(name)
            self.assertEqual(len(actual), self.model.config.num_hidden_layers)
            with torch.no_grad():
                fresh = self.model.model(self.prefixes[name], use_cache=True, return_dict=True)
            for (key, value), layer, fresh_layer in zip(actual,
                    self.oracles[name].cache.layers, fresh.past_key_values.layers):
                torch.testing.assert_close(key, layer.keys, atol=2e-6, rtol=1e-5)
                torch.testing.assert_close(value, layer.values, atol=2e-6, rtol=1e-5)
                torch.testing.assert_close(key, fresh_layer.keys, atol=2e-6, rtol=1e-5)
                torch.testing.assert_close(value, fresh_layer.values, atol=2e-6, rtol=1e-5)

    def run_chunks(self, chunks):
        calls = []
        hook = self.model.model.register_forward_pre_hook(
            lambda _m, _args, kwargs: calls.append((kwargs['input_ids'].shape,
                kwargs['position_ids'].clone(), kwargs['attention_mask']['full_attention'].shape)),
            with_kwargs=True)
        try:
            packed = self.target.append(chunks)
        finally:
            hook.remove()
        q = sum(chunk.shape[1] for chunk in chunks.values())
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], (1, q))
        self.assertEqual(calls[0][2], (1, 1, q, sum(self.target.lengths.values())))
        expected_positions = torch.cat([torch.arange(self.prefixes[name].shape[1],
            self.prefixes[name].shape[1]+chunk.shape[1]) for name, chunk in chunks.items()])[None]
        torch.testing.assert_close(calls[0][1], expected_positions, atol=0, rtol=0)
        all_logits = self.target.predict(packed)
        last_logits = self.target.predict(packed, last_only=True)
        for name, chunk in chunks.items():
            start = self.prefixes[name].shape[1]
            oracle = self.oracles[name]
            solo = oracle.prefill(chunk) if not start else oracle.append(chunk)
            self.prefixes[name] = torch.cat((self.prefixes[name], chunk), dim=1)
            actual = packed.for_request(name)
            self.assertEqual((actual.start, actual.end), (start, self.prefixes[name].shape[1]))
            torch.testing.assert_close(actual.last, solo.last, atol=2e-6, rtol=1e-5)
            torch.testing.assert_close(actual.context, solo.context, atol=2e-6, rtol=1e-5)
            torch.testing.assert_close(all_logits[name], oracle.predict(solo), atol=2e-6, rtol=1e-5)
            torch.testing.assert_close(last_logits[name], all_logits[name][:, -1], atol=0, rtol=0)
            # Independent fresh full recomputation, not only another mutable cache.
            with torch.no_grad():
                fresh = self.model.model(self.prefixes[name], use_cache=False, return_dict=True)
            torch.testing.assert_close(actual.last, fresh.last_hidden_state[:, start:], atol=2e-6, rtol=1e-5)
        self.assert_kv()
        self.assertFalse(packed.last.requires_grad)
        return packed

    def crop(self, name, length):
        self.target.crop(name, length)
        self.oracles[name].crop(length)
        self.prefixes[name] = self.prefixes[name][:, :length]

    def remove(self, name):
        self.target.remove_request(name)
        del self.prefixes[name], self.oracles[name]

    def test_mixed_lengths_one_forward_and_explicit_dense_work(self):
        for name in ('a', 'b', 'c'):
            self.add(name)
        first = self.run_chunks({'a': ids(1, 2, 3), 'b': ids(4), 'c': ids(5, 6, 7, 8)})
        self.assertEqual(first.work['physical_query_tokens'], 8)  # not 3 * 4
        self.assertEqual(first.work['dense_attention_pairs_per_head_layer'], 64)
        self.assertEqual(first.work['allowed_causal_pairs'], 6+1+10)
        self.assertEqual(first.work['masked_cross_request_pairs'], 64-(9+1+16))
        second = self.run_chunks({'b': ids(9, 10, 11), 'a': ids(12)})
        self.assertEqual(second.work['physical_query_tokens'], 4)
        self.assertEqual(second.work['physical_kv_tokens'], 12)
        self.assertEqual(second.work['dense_attention_pairs_per_head_layer'], 48)
        self.assertEqual(second.work['allowed_causal_pairs'], (3+6)+(3+1))
        self.assertEqual(second.work['masked_cross_request_pairs'], 48-(3*4+1*4))

    def test_independent_all_reject_partial_crop_exit_and_new_requests(self):
        for name in ('a', 'b', 'eos'):
            self.add(name)
        self.run_chunks({'a': ids(1,2,3), 'b': ids(4,5), 'eos': ids(6)})
        self.run_chunks({'a': ids(11,12,13), 'b': ids(14,15,16), 'eos': ids(17,18)})
        self.crop('a', 3)  # reject all of its just-verified block
        self.crop('b', 3)  # keep one of three candidates
        eos_kv = self.target.request_kv('eos')
        self.assert_kv()
        self.remove('eos')  # caller has committed EOS and releases the request
        self.add('new')
        self.add('eos')  # reused external ID is a fresh request, with a new marker
        self.run_chunks({'new': ids(41,42), 'a': ids(43), 'eos': ids(44), 'b': ids(45,46)})
        self.assertEqual(self.target.lengths, {'a':4, 'b':5, 'new':2, 'eos':1})
        self.assertEqual(eos_kv[0][0].shape[-2], 3)  # returned audit copies survive removal
        for name in tuple(self.prefixes):
            self.crop(name, 0)
        self.assertEqual(self.target.cache.get_seq_length(), 0)
        self.run_chunks({'b':ids(50), 'new':ids(51,52)})

    def test_every_crop_boundary_with_interleaved_cache(self):
        for keep in range(6):
            with self.subTest(keep=keep):
                self.setUp()
                self.add('a'); self.add('b')
                self.run_chunks({'a':ids(1,2), 'b':ids(3,4,5)})
                self.run_chunks({'b':ids(6), 'a':ids(7,8,9)})
                self.crop('a',keep)
                self.run_chunks({'a':ids(40,41), 'b':ids(42)})

    def test_cross_request_and_future_token_poison_isolation(self):
        outputs = []
        for poison in (False, True):
            target = PackedTarget(self.model, (0,1))
            target.add_request('a'); target.add_request('b')
            target.append({'a':ids(1,2), 'b':ids(3,4)})
            if poison:
                selected = target.key_requests == target._markers['b']
                for layer in target.cache.layers:
                    layer.keys[:,:,selected] = 1000
                    layer.values[:,:,selected] = -1000
            batch = target.append({'b':ids(55,56) if poison else ids(5,6),
                                   'a':ids(7,63) if poison else ids(7,8)})
            # A's first new query must see neither B nor A's later changed token.
            outputs.append((batch.for_request('a').last[:,:1],
                            batch.for_request('a').context[:,:1], target.predict(batch)['a'][:,:1]))
        for actual, expected in zip(*outputs):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_invalid_inputs_preserve_cache_and_failure_resets_all(self):
        self.add('a'); self.run_chunks({'a':ids(1,2)})
        before = self.target.request_kv('a')
        for bad in ({}, {'missing':ids(1)}, {'a':ids()}, {'a':ids(64)}, {'a':ids(1).float()}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.target.append(bad)
            self.assertEqual(self.target.lengths, {'a':2})
        for length in (-1, 3, True, 1.5):
            with self.assertRaises(ValueError): self.target.crop('a',length)
        with self.assertRaises(ValueError): self.target.add_request('a')
        for actual, expected in zip(self.target.request_kv('a'), before):
            for x,y in zip(actual,expected): torch.testing.assert_close(x,y,atol=0,rtol=0)
        def fail(_module, _args):
            raise RuntimeError('injected mid-forward failure')
        hook = self.model.model.layers[1].register_forward_pre_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError,'injected'):
                self.target.append({'a':ids(3)})
        finally:
            hook.remove()
        self.assertEqual(self.target.lengths,{})
        self.assertEqual(self.target.cache.get_seq_length(),0)
        self.target.add_request('fresh')
        self.assertEqual(self.target.append({'fresh':ids(4)}).last.shape[1],1)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
