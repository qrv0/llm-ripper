import torch.nn as nn

from llm_ripper.utils.architecture import replace_layer_submodule


class Leaf(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.lin(x)


class GPT2Like(nn.Module):
    def __init__(self):
        super().__init__()

        class C:
            model_type = "gpt2"

        self.config = C()

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Leaf()
                self.mlp = Leaf()

        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([Block()])


class LLaMALike(nn.Module):
    def __init__(self):
        super().__init__()

        class C:
            model_type = "llama"

        self.config = C()

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Leaf()
                self.mlp = Leaf()

        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block()])


def test_resolver_gpt2_and_llama():
    m1 = GPT2Like()
    ok_attn = replace_layer_submodule(
        m1, 0, "attn", Leaf(), model_type=m1.config.model_type
    )
    ok_ffn = replace_layer_submodule(
        m1, 0, "ffn", Leaf(), model_type=m1.config.model_type
    )
    assert ok_attn and ok_ffn

    m2 = LLaMALike()
    ok_attn2 = replace_layer_submodule(
        m2, 0, "attn", Leaf(), model_type=m2.config.model_type
    )
    ok_ffn2 = replace_layer_submodule(
        m2, 0, "ffn", Leaf(), model_type=m2.config.model_type
    )
    assert ok_attn2 and ok_ffn2
