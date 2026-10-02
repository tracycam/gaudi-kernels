"""Diagnostic tensors attributed to the calling layer, including shared RoPE.

Clones perturb scheduling: never use a run with these hooks for speed claims.
"""
import re
import torch

def install_stage_hooks(model, tensors, wanted):
    wanted = set(map(str, wanted))
    stack = []
    handles = []
    aliases = {}
    modules = {}
    for (name, module) in model.named_modules(remove_duplicate=False):
        aliases.setdefault(id(module), []).append(name)
        modules[id(module)] = module
        match = re.search('(?:^|\\.)layers\\.(\\d+)$', name)
        if match:

            def enter(module, args, name=name):
                stack.append(name)

            def leave(module, args, output):
                stack.pop()
            handles.append(module.register_forward_pre_hook(enter))
            handles.append(module.register_forward_hook(leave))

    def active_name(module):
        names = aliases[id(module)]
        name = next((n for n in names if stack and n.startswith(stack[-1] + '.')), None)
        if name is None:
            return None
        match = re.search('(?:^|\\.)layers\\.(\\d+)\\.', name)
        return name if match and match[1] in wanted else None

    def save(name, value):
        if name and isinstance(value, torch.Tensor):
            tensors[name, tuple(value.shape)] = value.clone()
    suffixes = ('input_layernorm', 'qkv_proj', 'rotary_emb', 'attn', 'o_proj', 'post_attention_layernorm', 'gate_up_proj', 'down_proj', 'mlp', 'gate', 'experts')
    quant_methods = {}
    for (identity, module) in modules.items():
        if any((n.endswith(suffixes) for n in aliases[identity])):

            def hook(module, args, output):
                name = active_name(module)
                if not name:
                    return
                if name.endswith('rotary_emb') and isinstance(output, tuple):
                    save(name + '.q', output[0])
                    save(name + '.k', output[1])
                else:
                    save(name, output[0] if isinstance(output, tuple) else output)
            handles.append(module.register_forward_hook(hook))
        if type(module).__name__ == 'NativeExpertTP':

            def routes(module, args, kwargs):
                name = active_name(module)
                if name:
                    save(name + '.route_ids', args[1] if len(args) > 1 else kwargs['topk_ids'])
                    save(name + '.route_weights', args[2] if len(args) > 2 else kwargs['topk_weights'])
            handles.append(module.register_forward_pre_hook(routes, with_kwargs=True))
        if any((n.endswith(('o_proj', 'down_proj')) for n in aliases[identity])) and hasattr(module, 'quant_method'):
            quant_methods[id(module.quant_method)] = module.quant_method
    for quant in quant_methods.values():
        original = quant.apply

        def apply(owner, *args, _original=original, **kwargs):
            value = _original(owner, *args, **kwargs)
            name = active_name(owner)
            if name and name.endswith(('o_proj', 'down_proj')):
                save(name + '.local', value)
            return value
        quant.apply = apply
    return handles
