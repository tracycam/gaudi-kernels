"""Bounded decode-only stage capture for locating error amplification.

These hooks add device clones and CPU copies, so runs containing them are
diagnostic only. They are armed after performance phases and removed afterward.
The original operator outputs, routing IDs, order and arithmetic are unchanged.
"""
import re

class LayerDiagnostics:

    def __init__(self, model, *, capture_rotary=True):
        self.enabled = False
        self.capture_rotary = capture_rotary
        self.stack = []
        self.values = {}
        self.handles = []
        self.aliases = {}
        modules = {}
        for (name, module) in model.named_modules(remove_duplicate=False):
            self.aliases.setdefault(id(module), []).append(name)
            modules[id(module)] = module
            if re.search('(?:^|\\.)layers\\.\\d+$', name):

                def enter(module, args, kwargs, name=name):
                    if self.enabled:
                        self.stack.append(name)
                        value = kwargs.get('hidden_states', args[1] if len(args) > 1 else None)
                        self.save(name + '.input', value)

                def leave(module, args, output, name=name):
                    if self.enabled:
                        if isinstance(output, tuple):
                            self.save(name + '.output', output[0])
                            self.save(name + '.residual', output[1])
                        self.stack.pop()
                self.handles.append(module.register_forward_pre_hook(enter, with_kwargs=True))
                self.handles.append(module.register_forward_hook(leave))
        suffixes = ('input_layernorm', 'qkv_proj', 'rotary_emb', 'attn', 'o_proj', 'post_attention_layernorm', 'mlp', 'gate', 'experts')
        if not capture_rotary:
            suffixes = tuple((s for s in suffixes if s != 'rotary_emb'))
        for (identity, module) in modules.items():
            if any((name.endswith(suffixes) for name in self.aliases[identity])):

                def hook(module, args, output):
                    name = self.name(module)
                    if name:
                        if name.endswith('rotary_emb') and isinstance(output, tuple):
                            self.save(name + '.q', output[0])
                            self.save(name + '.k', output[1])
                        else:
                            self.save(name, output[0] if isinstance(output, tuple) else output)
                self.handles.append(module.register_forward_hook(hook))
            if type(module).__name__ == 'NativeExpertTP':

                def routes(module, args, kwargs):
                    name = self.name(module)
                    if name:
                        self.save(name + '.route_ids', args[1] if len(args) > 1 else kwargs['topk_ids'])
                        self.save(name + '.route_weights', args[2] if len(args) > 2 else kwargs['topk_weights'])
                self.handles.append(module.register_forward_pre_hook(routes, with_kwargs=True))

    def name(self, module):
        if not self.enabled or not self.stack:
            return None
        return next((name for name in self.aliases[id(module)] if name.startswith(self.stack[-1] + '.')), None)

    def save(self, name, value):
        import torch
        if isinstance(value, torch.Tensor):
            if value.numel() > 65536:
                raise ValueError('stage capture exceeded decode-only tensor budget')
            self.values[name] = value.detach().clone()

    def flush(self, path, token_ids, positions):
        import torch
        import habana_frameworks.torch.core as hc
        hc.mark_step()
        torch.hpu.synchronize()
        torch.save({'stages': {name: value.cpu() for (name, value) in self.values.items()}, 'input_ids': token_ids.cpu(), 'positions': positions.cpu(), 'schema': {'version': 2, 'rotary_module_outputs': self.capture_rotary}, 'scope': 'intrusive decode-only stage diagnostic; excluded from TPS'}, path)
        self.values.clear()

    def close(self):
        self.enabled = False
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.values.clear()
        self.stack.clear()
