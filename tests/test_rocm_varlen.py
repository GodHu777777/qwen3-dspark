"""CPU contracts for explicit private backend; GPU calls are forbidden or substituted."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, PropertyMock, patch

import torch
from dspark_qwen.rocm_varlen import BACKEND, RUNTIME_PIN, PinnedRocmVarlenKernel, validate_runtime
from dspark_qwen.varlen_target import PackedLayout

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    q=torch.zeros(9,16,128,dtype=torch.bfloat16)
    k=torch.zeros(46,8,128,dtype=torch.bfloat16)
    layout=PackedLayout((1,2),(1,8),(17,29),torch.tensor([0,1,9],dtype=torch.int32),
        torch.tensor([0,17,46],dtype=torch.int32),torch.arange(46),46)
    return q,k,k.clone(),layout


class PinnedContractTests(unittest.TestCase):
    def test_every_runtime_pin_field_is_fail_closed(self):
        validate_runtime(dict(RUNTIME_PIN))
        for key in RUNTIME_PIN:
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError,'runtime mismatch'):
                validate_runtime(dict(RUNTIME_PIN,**{key:'changed'}))
        with self.assertRaises(RuntimeError): validate_runtime({})

    def test_cpu_constructor_has_no_fallback(self):
        with self.assertRaisesRegex(RuntimeError,'ROCm GPU'):
            PinnedRocmVarlenKernel('cpu')

    def make_kernel(self):
        with patch('dspark_qwen.rocm_varlen.runtime_identity',return_value=dict(RUNTIME_PIN)) as check:
            kernel=PinnedRocmVarlenKernel('cpu')
        check.assert_called_once()
        kernel._operator=Mock(side_effect=lambda q,*a,**kw:(q.clone(),))
        return kernel

    def test_exact_entry_arguments_runtime_cached_and_layout_mutation_refused(self):
        q,k,v,layout=fixture(); kernel=self.make_kernel()
        with patch.object(torch.Tensor,'is_cuda',new_callable=PropertyMock,return_value=True):
            kernel(q,k,v,layout,scale=128**-.5)
            kernel(q,k,v,layout,scale=128**-.5)
            self.assertEqual(kernel._operator.call_count,2)
            args,kwargs=kernel._operator.call_args
            self.assertEqual(args[5:],(8,29,0.,True,False))
            self.assertEqual(kwargs,dict(scale=128**-.5,window_size_left=None,window_size_right=None,
                seqused_k=None,alibi_slopes=None,block_table=None,num_splits=None))
            self.assertIs(kernel._validated_layout,layout)
            layout.cu_query[1]=2
            with self.assertRaisesRegex(ValueError,'Cumulative lengths'):
                kernel(q,k,v,layout,scale=128**-.5)
        self.assertEqual(kernel._operator.call_count,2)

    def test_normal_no_grad_and_inference_layouts_validate_and_detect_changes(self):
        for mode in [contextlib.nullcontext,torch.no_grad,torch.inference_mode]:
            with self.subTest(mode=mode.__name__),mode():
                q,k,v,layout=fixture();kernel=self.make_kernel()
                with patch.object(torch.Tensor,'is_cuda',new_callable=PropertyMock,return_value=True):
                    kernel(q,k,v,layout,scale=128**-.5)
                    kernel(q,k,v,layout,scale=128**-.5)
                    layout.cu_key[1]=18
                    with self.assertRaisesRegex(ValueError,'Cumulative lengths'):
                        kernel(q,k,v,layout,scale=128**-.5)
                self.assertEqual(kernel._operator.call_count,2)

    def test_bad_tensor_shape_dtype_scale_and_metadata_fail_before_operator(self):
        for mutation in ['dtype','heads','scale','cu_dtype','noncontiguous','tokens','requires_grad']:
            q,k,v,layout=fixture();kernel=self.make_kernel();scale=128**-.5
            if mutation=='dtype': q=q.float()
            if mutation=='requires_grad':q.requires_grad_(True)
            if mutation=='heads': q=q[:,:8].contiguous()
            if mutation=='scale': scale=1.
            if mutation=='cu_dtype': object.__setattr__(layout,'cu_query',layout.cu_query.long())
            if mutation=='noncontiguous': q=torch.zeros(9,16,256,dtype=torch.bfloat16)[:,:,::2]
            if mutation=='tokens': q=q[:1]
            with self.subTest(mutation=mutation), patch.object(torch.Tensor,'is_cuda',new_callable=PropertyMock,return_value=True):
                with self.assertRaises(ValueError): kernel(q,k,v,layout,scale=scale)
            kernel._operator.assert_not_called()

    def test_operator_exception_propagates_without_public_fallback(self):
        kernel=self.make_kernel();kernel._operator.side_effect=RuntimeError('native fixture failure')
        with patch.object(torch.Tensor,'is_cuda',new_callable=PropertyMock,return_value=True), patch('dspark_qwen.varlen_target.native_varlen') as public:
            with self.assertRaisesRegex(RuntimeError,'native fixture failure'):
                kernel(*fixture(),scale=128**-.5)
            public.assert_not_called()

    def test_target_selection_is_explicit_and_test_injection_exclusive(self):
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from dspark_qwen.varlen_target import VarlenPackedTarget,native_varlen,test_only_dense_varlen
        cfg=Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8)
        cfg._attn_implementation='sdpa'
        target=VarlenPackedTarget(Qwen3ForCausalLM(cfg))
        self.assertIs(target._varlen_kernel,native_varlen)
        self.assertEqual(target._backend_label,'native_varlen_unverified')
        with self.assertRaisesRegex(ValueError,'mutually exclusive'):
            VarlenPackedTarget(None,native_backend=BACKEND,test_kernel=test_only_dense_varlen)
        with self.assertRaisesRegex(ValueError,'Unknown explicit'):
            VarlenPackedTarget(None,native_backend='auto')


class PinnedGateTests(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('probe_review',ROOT/'scripts/probe_varlen.py')
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)

    def test_public_protocol_is_unchanged_and_private_identity_is_explicit(self):
        public=self.module.protocol_for_backend()
        h=hashlib.sha256(json.dumps(public,sort_keys=True).encode()).hexdigest()
        self.assertEqual(h,'08c501349bb59d30a888f1a5669a6c5bcb0bdfbf5f9e894c409fb2e702d49ff9')
        private=self.module.protocol_for_backend(BACKEND)
        self.assertEqual(private['original_public_protocol_sha256'],h)
        self.assertEqual(private['cases'],public['cases'])
        for key in ['atol','rtol','max_rms_error','seed','timeout_seconds','checks','oracles']:
            self.assertEqual(private[key],public[key])
        self.assertEqual(private['runtime_pin'],RUNTIME_PIN)
        self.assertEqual(private['window_size'],[None,None])
        self.assertNotEqual(hashlib.sha256(json.dumps(private,sort_keys=True).encode()).hexdigest(),h)

    @contextlib.contextmanager
    def cpu_worker(self,fail=False):
        original_device=torch.device
        counts=[]
        def device(value,*args,**kwargs):
            if isinstance(value,str) and value.startswith('cuda'): return original_device('cpu')
            return original_device(value,*args,**kwargs)
        def flash(q,k,v,cuq,cuk,*args,**kwargs):
            counts.append(1)
            if fail: return (torch.zeros_like(q),)
            qs,ks=cuq.tolist(),cuk.tolist();rows=[]
            for qa,qb,ka,kb in zip(qs,qs[1:],ks,ks[1:]):
                nq,nk=qb-qa,kb-ka
                mask=torch.arange(nk)[None,:] <= torch.arange(nk-nq,nk)[:,None]
                rows.append(torch.nn.functional.scaled_dot_product_attention(
                    q[qa:qb].transpose(0,1)[None].float(),k[ka:kb].transpose(0,1)[None].float(),
                    v[ka:kb].transpose(0,1)[None].float(),attn_mask=mask[None,None],
                    dropout_p=0.,scale=kwargs['scale'],enable_gqa=True).squeeze(0).transpose(0,1).to(q.dtype))
            return (torch.cat(rows),)
        class Profiler:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def key_averages(self):return [SimpleNamespace(key='aten::_flash_attention_forward',count=1)]
        with contextlib.ExitStack() as stack:
            for name,value in [('is_available',True),('get_device_name','CPU substitute'),
                               ('get_device_properties',SimpleNamespace(gcnArchName='CPU')),('max_memory_allocated',0)]:
                stack.enter_context(patch.object(torch.cuda,name,return_value=value))
            for name in ['set_device','reset_peak_memory_stats','synchronize']:
                stack.enter_context(patch.object(torch.cuda,name))
            stack.enter_context(patch.object(torch,'device',side_effect=device))
            stack.enter_context(patch.object(torch.Tensor,'is_cuda',new_callable=PropertyMock,return_value=True))
            stack.enter_context(patch.object(torch.profiler,'profile',side_effect=lambda **kwargs:Profiler()))
            stack.enter_context(patch.object(torch.ops.aten,'_flash_attention_forward',side_effect=flash))
            stack.enter_context(patch('dspark_qwen.rocm_varlen.runtime_identity',return_value=dict(RUNTIME_PIN)))
            stack.enter_context(patch.object(torch.backends.cuda,'preferred_rocm_fa_library',return_value='CPU substitute'))
            yield counts

    def test_full_tensor_gate_control_flow_uses_all_original_cases_on_cpu_substitute(self):
        with tempfile.TemporaryDirectory() as tmp,self.cpu_worker() as calls:
            out=Path(tmp);self.module.worker(out,BACKEND)
            result=json.loads((out/'worker-result.json').read_text())
            self.assertEqual(result['status'],'passed')
            self.assertEqual(len(calls),11)
            self.assertEqual(len(result['cases']),3)
            self.assertEqual(len(result['native_calls']),11)
            self.assertTrue(all((out/r['tensor_file']).exists() for r in result['native_calls']))
            self.assertTrue(all(r['other_request_isolation_exact'] for r in result['cases']))
            self.assertEqual(result['protocol']['backend'],BACKEND)

    def test_numerical_failure_keeps_raw_output_oracles_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp,self.cpu_worker(fail=True) as calls:
            out=Path(tmp)
            with self.assertRaises(AssertionError):self.module.worker(out,BACKEND)
            result=json.loads((out/'worker-result.json').read_text())
            self.assertEqual(len(calls),1)
            self.assertEqual(result['native_calls'][0]['status'],'raw_output_saved')
            self.assertEqual(len(result['comparisons']),1)
            self.assertTrue((out/'private-oracles-case0.pt').exists())
            tensors=torch.load(out/'private-native-000.pt',weights_only=True)
            self.assertIn('output',tensors)


if __name__=='__main__':unittest.main()
