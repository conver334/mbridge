
"""
AutoBridge wrapper for seamless integration with Megatron-Bridge.
"""

from typing import Optional, Iterator, Tuple, Any, List, Callable
import torch
from transformers import PretrainedConfig

try:
    from megatron.bridge import AutoBridge as MegatronAutoBridge
    from megatron.core import parallel_state
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
    from megatron.core.transformer.module import MegatronModule
except ImportError:
    raise ImportError(
        "Megatron-Bridge is required. Please install it from github.com/NVIDIA-NeMo/Megatron-Bridge"
    )


class AutoBridge:
    """
    AutoBridge wrapper for Megatron-Bridge integration.
    
    Example:
        >>> bridge = AutoBridge.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> model = bridge.get_model()
        >>> bridge.load_weights(model, hf_model_path)
    """
    
    def __init__(self, megatron_bridge_instance):
        self._bridge = megatron_bridge_instance
        self._extra_args = {}
        self._hf_config = None
        self._provider = None
        
        if hasattr(self._bridge, 'hf_pretrained'):
            hf_pretrained = self._bridge.hf_pretrained
            if hasattr(hf_pretrained, 'config'):
                self._hf_config = hf_pretrained.config
            elif isinstance(hf_pretrained, PretrainedConfig):
                self._hf_config = hf_pretrained
    
    @classmethod
    def from_config(cls, hf_config: PretrainedConfig, **kwargs) -> "AutoBridge":
        megatron_bridge = MegatronAutoBridge.from_hf_config(hf_config)
        wrapper = cls(megatron_bridge)
        wrapper._hf_config = hf_config
        return wrapper
    
    @classmethod
    def from_pretrained(cls, hf_model_path: str, trust_remote_code: bool = False, **kwargs) -> "AutoBridge":
        megatron_bridge = MegatronAutoBridge.from_hf_pretrained(
            hf_model_path,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        return cls(megatron_bridge)
    
    @property
    def config(self) -> TransformerConfig:
        if not hasattr(self._bridge, '_transformer_config') or self._bridge._transformer_config is None:
            return self._bridge.transformer_config
        return self._bridge._transformer_config
    
    @property
    def hf_config(self) -> PretrainedConfig:
        return self._hf_config
    
    def set_extra_args(self, **kwargs):
        self._extra_args.update(kwargs)
        if hasattr(self._bridge, '_transformer_config'):
            current_config = self._bridge._transformer_config
            if current_config is not None:
                config_dict = current_config.as_dict() if hasattr(current_config, 'as_dict') else {}
                config_dict.update(kwargs)
                
                try:
                    for key, value in kwargs.items():
                        if hasattr(current_config, key):
                            setattr(current_config, key, value)
                except Exception as e:
                    print(f"Warning: Failed to update transformer config: {e}")
    
    def _create_pre_wrap_hook_adapter(self, callbacks: List[Callable]) -> Callable:
        def pre_wrap_hook_adapter(model_list):
            config = self.config
            hf_config = self.hf_config
            
            for model in model_list:
                pre_process = parallel_state.is_pipeline_first_stage()
                post_process = parallel_state.is_pipeline_last_stage()
                
                for callback in callbacks:
                    try:
                        callback(
                            model,
                            pre_process=pre_process,
                            post_process=post_process,
                            config=config,
                            hf_config=hf_config,
                        )
                        print(f"Successfully executed callback: {callback.__name__}")
                    except Exception as e:
                        print(f"Warning: Failed to execute callback {callback.__name__}: {e}")
            
            return model_list
        
        return pre_wrap_hook_adapter
    
    def get_model(self, post_model_creation_callbacks: Optional[List[Callable]] = None, **kwargs) -> List:
        if post_model_creation_callbacks is None:
            post_model_creation_callbacks = []
        
        self._provider = self._bridge.to_megatron_provider(load_weights=False)
        
        if self._extra_args:
            for key, value in self._extra_args.items():
                if hasattr(self._provider, key):
                    setattr(self._provider, key, value)
        
        from megatron.core import parallel_state as mpu
        self._provider.tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()
        self._provider.pipeline_model_parallel_size = mpu.get_pipeline_model_parallel_world_size()
        self._provider.expert_model_parallel_size = mpu.get_expert_model_parallel_world_size()
        self._provider.expert_tensor_parallel_size = mpu.get_expert_tensor_parallel_world_size()
        
        if post_model_creation_callbacks:
            pre_wrap_hook = self._create_pre_wrap_hook_adapter(post_model_creation_callbacks)
            self._provider.register_pre_wrap_hook(pre_wrap_hook)
            print(f"Registered {len(post_model_creation_callbacks)} post_model_creation_callbacks as pre_wrap_hook")
        
        if hasattr(self._provider, "finalize"):
            print("Finalizing provider...")
            self._provider.finalize()
            
        print("Provider config:")
        print(f"  tensor_model_parallel_size: {self._provider.tensor_model_parallel_size}")
        print(f"  pipeline_model_parallel_size: {self._provider.pipeline_model_parallel_size}")
        print(f"  expert_model_parallel_size: {self._provider.expert_model_parallel_size}")
        print(f"  expert_tensor_parallel_size: {self._provider.expert_tensor_parallel_size}")
        
        model = self._provider.provide_distributed_model(wrap_with_ddp=False)
        
        return model if isinstance(model, list) else [model]
    
    def load_weights(self, model, hf_model_path: str, show_progress: bool = False):
        pre_trained = PreTrainedCausalLM.from_pretrained(hf_model_path)
        self._bridge._model_bridge.load_weights_hf_to_megatron(pre_trained, model)
    
    def export_weights(
        self, 
        model, 
        show_progress: bool = False,
        cpu: bool = False
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        if isinstance(model, list):
            actual_model = model
        else:
            actual_model = [model]
        
        try:
            for name, tensor in self._bridge.export_hf_weights(actual_model, cpu=cpu):
                yield name, tensor
        except Exception as e:
            print(f"Warning: Failed to export weights using Megatron-Bridge API: {e}")
            raise
    
    def save_weights(
        self, 
        model, 
        save_path: str, 
        memory_efficient: bool = False,
        show_progress: bool = False
    ):
        if isinstance(model, list):
            actual_model = model
        else:
            actual_model = [model]
        
        try:
            self._bridge.save_hf_pretrained(actual_model, save_path)
        except Exception as e:
            print(f"Warning: Failed to save weights using Megatron-Bridge API: {e}")
            raise
    
    @classmethod
    def list_supported_models(cls) -> List[str]:
        try:
            return MegatronAutoBridge.list_supported_models()
        except Exception as e:
            print(f"Warning: Failed to list supported models: {e}")
            return []


__all__ = ["AutoBridge"]


