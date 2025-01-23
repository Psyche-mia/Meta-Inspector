import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np
from Meta_Inspector.AnomalyCLIP_lib.simple_tokenizer import SimpleTokenizer as _Tokenizer
# from open_clip import tokenizer
# simple_tokenizer = tokenizer.SimpleTokenizer()
from copy import deepcopy
import torch.nn as nn

_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

def encode_text_with_prompt_ensemble(model, texts, device):
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.', 'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.', 'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']

    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(texts[0]) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence)
        class_embeddings = model.encode_text(prompted_sentence.to(device))
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        text_features.append(class_embedding)

    text_features = torch.stack(text_features, dim=1).to(device).t()

    return text_features



def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])
class AnomalyCLIP_PromptLearner(nn.Module):
    def __init__(self, clip_model, design_details):
        super().__init__()
        classnames = ["object"]
        self.n_cls = len(classnames)
        self.n_ctx = design_details["Prompt_length"]
        n_ctx_pos = self.n_ctx
        n_ctx_neg = self.n_ctx
        self.text_encoder_n_ctx = design_details["learnable_text_embedding_length"] 
        ctx_init_pos = ""
        ctx_init_neg = ""
        dtype = clip_model.transformer.get_cast_dtype()

        ctx_dim = clip_model.ln_final.weight.shape[0]

        
        self.classnames = classnames

        self.state_normal_list = [
            "{}",
        ]

        self.state_anomaly_list = [
            "damaged {}",
        ]
        
        normal_num = len(self.state_normal_list)
        anormaly_num = len(self.state_anomaly_list)
        self.normal_num = normal_num
        self.anormaly_num = anormaly_num

        if ctx_init_pos and ctx_init_neg:
            # use given words to initialize context vectors
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            n_ctx_neg = len(ctx_init_neg.split(" "))
            #初始化text成bpd编码
            prompt_pos = tokenize(ctx_init_pos)
            prompt_neg = tokenize(ctx_init_neg)
            with torch.no_grad():
                #生成相应的text embedding
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            #这些是去除出来EOS 和 # CLS, EOS， 获得可学习的textual prompt
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            if True:
                ctx_vectors_pos_ = []
                ctx_vectors_neg_ = []
                for _ in range(self.n_cls):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                    ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)

        else:
            # Random Initialization
            if True:
                print("Initializing class-specific contexts")
                #这里是cls是类的个数，n_ctx_pos代表learnable token的长度，ctx_dim表示prompt的dimension
                ctx_vectors_pos = torch.empty(self.n_cls, self.normal_num, n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(self.n_cls, self.anormaly_num, n_ctx_neg, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)
        self.compound_prompts_depth = design_details["learnable_text_embedding_depth"]
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            print("single_para", single_para.shape)
            nn.init.normal_(single_para, std=0.02)

        single_layer = nn.Linear(ctx_dim, 896)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)


        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        
        prompts_pos = [prompt_prefix_pos +  " " + template.format(name)+ "." for template in self.state_normal_list for name in classnames]
        prompts_neg = [prompt_prefix_neg +  " " + template.format(name)+ "." for template in self.state_anomaly_list for name in classnames]

        # Tokenize prompts
        tokenized_prompts_pos = []
        tokenized_prompts_neg = []
     
        for p_pos in prompts_pos:
            tokenized_prompts_pos.append(tokenize(p_pos))
        for p_neg in prompts_neg:
            tokenized_prompts_neg.append(tokenize(p_neg))
            
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)
        #生成相应的text embedding
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            n, l, d = embedding_pos.shape
            print("embedding_pos", embedding_pos.shape)
            embedding_pos = embedding_pos.reshape(normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_neg = embedding_neg.reshape(anormaly_num, self.n_cls, l, d).permute(1, 0, 2, 3)


        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :] )
        self.register_buffer("token_suffix_pos", embedding_pos[:, :,1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:,:, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + n_ctx_neg:, :])

        n, d = tokenized_prompts_pos.shape
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(normal_num, self.n_cls, d).permute(1, 0, 2)

        n, d = tokenized_prompts_neg.shape
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(anormaly_num, self.n_cls, d).permute(1, 0, 2)

        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        # tokenized_prompts = torch.cat([tokenized_prompts_pos, tokenized_prompts_neg], dim=0)  # torch.Tensor
        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)
        print("tokenized_prompts shape", self.tokenized_prompts_pos.shape, self.tokenized_prompts_neg.shape)



    def forward(self, cls_id =None):

        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg
        # print("self.ctx_neg in def forward", self.ctx_neg)
        # print("shape", self.ctx_pos[0:1].shape, ctx_pos.shape)
        prefix_pos = self.token_prefix_pos
        prefix_neg = self.token_prefix_neg
        suffix_pos = self.token_suffix_pos
        suffix_neg = self.token_suffix_neg

        # print(prefix_pos.shape, prefix_neg.shape)
        print(f"prefix shape: {prefix_pos.shape}")
        print(f"ctx shape: {ctx_pos.shape}")
        print(f"suffix shape: {suffix_pos.shape}")

        prompts_pos = torch.cat(
            [
                # N(the number of template), 1, dim
                prefix_pos,  # (n_cls, 1, dim)
                ctx_pos,  # (n_cls, n_ctx, dim)
                suffix_pos,  # (n_cls, *, dim)
            ],
            dim=2,
        )

        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls, 1, dim)
                ctx_neg,  # (n_cls, n_ctx, dim)
                suffix_neg,  # (n_cls, *, dim)
            ],
            dim=2,
        )
        _, _, l, d = prompts_pos.shape
        prompts_pos = prompts_pos.reshape(-1, l, d)
        _, _, l, d = prompts_neg.shape
        prompts_neg = prompts_neg.reshape(-1, l, d)
        prompts = torch.cat([prompts_pos, prompts_neg], dim=0)


        _, l, d = self.tokenized_prompts_pos.shape
        tokenized_prompts_pos = self.tokenized_prompts_pos.reshape(-1,  d)
        _, l, d = self.tokenized_prompts_neg.shape
        tokenized_prompts_neg = self.tokenized_prompts_neg.reshape(-1,  d)
        tokenized_prompts = torch.cat((tokenized_prompts_pos, tokenized_prompts_neg), dim = 0)


        return prompts, tokenized_prompts, self.compound_prompts_text
    
class Custom_AnomalyCLIP_PromptLearner(nn.Module):
    def __init__(self, clip_model, design_details):
        super().__init__()
        classnames = ["object"]
        self.n_cls = len(classnames)
        self.n_ctx = design_details["Prompt_length"]
        
        # Initialize context vectors for normal, anomaly states, and anomaly subtypes
        n_ctx_pos = self.n_ctx
        n_ctx_neg = self.n_ctx
        n_ctx_anomaly_subtype = self.n_ctx
        
        self.text_encoder_n_ctx = design_details["learnable_text_embedding_length"]
        dtype = clip_model.transformer.get_cast_dtype()

        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.classnames = classnames

        self.state_normal_list = ["{}"]
        self.state_anomaly_list = ["damaged {}"]
        self.anomaly_subtype_list = [
            "Physical Damage {}",
            "Contamination {}",
            "Morphological Anomalies {}",
            "Surface Defects {}",
            "Manufacturing Defects {}"
        ]

        normal_num = len(self.state_normal_list)
        anomaly_num = len(self.state_anomaly_list)
        anomaly_subtype_num = len(self.anomaly_subtype_list)

        self.normal_num = normal_num
        self.anomaly_num = anomaly_num
        self.anomaly_subtype_num = anomaly_subtype_num
        # Initialize context vectors for normal, anomaly states, and anomaly subtypes
        ctx_init_pos = ""
        ctx_init_neg = ""
        ctx_init_anomaly_subtype = ""
        # Initialize context vectors for normal, anomaly states, and anomaly subtypes
        # self.ctx_pos, self.prompt_prefix_pos = self.initialize_context(clip_model, "", n_ctx_pos, normal_num, ctx_dim, dtype)
        # self.ctx_neg, self.prompt_prefix_neg = self.initialize_context(clip_model, "", n_ctx_neg, anomaly_num, ctx_dim, dtype)
        # self.ctx_anomaly_subtype, self.prompt_prefix_anomaly_subtype = self.initialize_context(clip_model, "", n_ctx_anomaly_subtype, anomaly_subtype_num, ctx_dim, dtype)

        # Ensure ctx_pos and ctx_neg are registered as parameters
        # self.ctx_pos = nn.Parameter(self.ctx_pos)
        # self.ctx_neg = nn.Parameter(self.ctx_neg)
        # self.ctx_anomaly_subtype = nn.Parameter(self.ctx_anomaly_subtype)
        if ctx_init_pos and ctx_init_neg:
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            n_ctx_neg = len(ctx_init_neg.split(" "))
            prompt_pos = tokenize(ctx_init_pos)
            prompt_neg = tokenize(ctx_init_neg)
            with torch.no_grad():
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            ctx_vectors_pos_ = []
            ctx_vectors_neg_ = []
            for _ in range(self.n_cls):
                ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
            ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
            ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)
        else:
            print("Initializing class-specific contexts")
            ctx_vectors_pos = torch.empty(self.n_cls, normal_num, n_ctx_pos, ctx_dim, dtype=dtype)
            ctx_vectors_neg = torch.empty(self.n_cls, anomaly_num, n_ctx_neg, ctx_dim, dtype=dtype)
            ctx_vectors_anomaly_subtype = torch.empty(self.n_cls, anomaly_subtype_num, n_ctx_anomaly_subtype, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            nn.init.normal_(ctx_vectors_anomaly_subtype, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)
            prompt_prefix_anomaly_subtype = " ".join(["X"] * n_ctx_anomaly_subtype)
        
        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized
        self.ctx_anomaly_subtype = nn.Parameter(ctx_vectors_anomaly_subtype)  # to be optimized
        
        # Initialize compound prompts for depth and text
        self.compound_prompts_depth = design_details["learnable_text_embedding_depth"]
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            print("single_para", single_para.shape)
            nn.init.normal_(single_para, std=0.02)

        single_layer = nn.Linear(ctx_dim, 896)
        self.compound_prompt_projections = self._get_clones(single_layer, self.compound_prompts_depth - 1)

        # Prepare tokenized prompts
        classnames = [name.replace("_", " ") for name in self.classnames]

        prompts_pos = [prompt_prefix_pos + " " + template.format(name) + "." for template in self.state_normal_list for name in classnames]
        prompts_neg = [prompt_prefix_neg + " " + template.format(name) + "." for template in self.state_anomaly_list for name in classnames]
        prompts_anomaly_subtype = [prompt_prefix_anomaly_subtype + " " + template.format(name) + "." for template in self.anomaly_subtype_list for name in classnames]

        tokenized_prompts_pos = [tokenize(p) for p in prompts_pos]
        tokenized_prompts_neg = [tokenize(p) for p in prompts_neg]
        tokenized_prompts_anomaly_subtype = [tokenize(p) for p in prompts_anomaly_subtype]

        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)
        tokenized_prompts_anomaly_subtype = torch.cat(tokenized_prompts_anomaly_subtype)

        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            embedding_anomaly_subtype = clip_model.token_embedding(tokenized_prompts_anomaly_subtype).type(dtype)

            # Check for NaNs in the embeddings
            if torch.isnan(embedding_pos).any() or torch.isnan(embedding_neg).any() or torch.isnan(embedding_anomaly_subtype).any():
                raise ValueError("NaN detected in token embeddings")

            n, l, d = embedding_pos.shape
            embedding_pos = embedding_pos.reshape(self.normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_neg = embedding_neg.reshape(self.anomaly_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_anomaly_subtype = embedding_anomaly_subtype.reshape(self.anomaly_subtype_num, self.n_cls, l, d).permute(1, 0, 2, 3)

        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :])
        self.register_buffer("token_suffix_pos", embedding_pos[:, :, 1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:, :, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + n_ctx_neg:, :])
        self.register_buffer("token_prefix_anomaly_subtype", embedding_anomaly_subtype[:, :, :1, :])
        self.register_buffer("token_suffix_anomaly_subtype", embedding_anomaly_subtype[:, :, 1 + n_ctx_anomaly_subtype:, :])

        n, d = tokenized_prompts_pos.shape
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(self.normal_num, self.n_cls, d).permute(1, 0, 2)
        n, d = tokenized_prompts_neg.shape
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(self.anomaly_num, self.n_cls, d).permute(1, 0, 2)
        n, d = tokenized_prompts_anomaly_subtype.shape
        tokenized_prompts_anomaly_subtype = tokenized_prompts_anomaly_subtype.reshape(self.anomaly_subtype_num, self.n_cls, d).permute(1, 0, 2)

        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        self.n_ctx_anomaly_subtype = n_ctx_anomaly_subtype
        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)
        self.register_buffer("tokenized_prompts_anomaly_subtype", tokenized_prompts_anomaly_subtype)
        print("tokenized_prompts shape", self.tokenized_prompts_pos.shape, self.tokenized_prompts_neg.shape, self.tokenized_prompts_anomaly_subtype.shape)

    # def initialize_context(self, clip_model, ctx_init, n_ctx, num_templates, ctx_dim, dtype):
    #     if ctx_init:
    #         ctx_init = ctx_init.replace("_", " ")
    #         n_ctx = len(ctx_init.split(" "))
    #         prompt = tokenize(ctx_init)
    #         with torch.no_grad():
    #             embedding = clip_model.token_embedding(prompt).type(dtype)
    #         ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
    #         prompt_prefix = ctx_init
    #         ctx_vectors = torch.stack([deepcopy(ctx_vectors) for _ in range(self.n_cls)], dim=0)
    #     else:
    #         print("Initializing context vectors")
    #         ctx_vectors = torch.empty(self.n_cls, num_templates, n_ctx, ctx_dim, dtype=dtype)
    #         nn.init.normal_(ctx_vectors, std=0.02)
    #         prompt_prefix = " ".join(["X"] * n_ctx)

    #     # Check for NaNs in the initialized context vectors
    #     if torch.isnan(ctx_vectors).any():
    #         raise ValueError("NaN detected in context vectors during initialization")
        
    #     return nn.Parameter(ctx_vectors), prompt_prefix

    def _get_clones(self, module, N):
        return nn.ModuleList([deepcopy(module) for _ in range(N)])

    def forward(self, cls_id=None):
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg
        ctx_anomaly_subtype = self.ctx_anomaly_subtype

        prefix_pos = self.token_prefix_pos
        prefix_neg = self.token_prefix_neg
        prefix_anomaly_subtype = self.token_prefix_anomaly_subtype

        suffix_pos = self.token_suffix_pos
        suffix_neg = self.token_suffix_neg
        suffix_anomaly_subtype = self.token_suffix_anomaly_subtype

        # Debugging 信息
        # print("self.ctx_neg in def forward", self.ctx_neg)
        # print("ctx_neg after assignment from self.ctx_neg:", ctx_neg)

        if torch.isnan(self.ctx_neg).any():
            raise ValueError("NaN detected in self.ctx_neg before assignment")

        if torch.isnan(ctx_neg).any():
            raise ValueError("NaN detected in ctx_neg after assignment")

        if torch.isnan(ctx_pos).any():
            raise ValueError("NaN detected in ctx_pos")
        if torch.isnan(ctx_anomaly_subtype).any():
            raise ValueError("NaN detected in ctx_anomaly_subtype")

        if torch.isnan(prefix_pos).any():
            raise ValueError("NaN detected in prefix_pos")
        if torch.isnan(prefix_neg).any():
            raise ValueError("NaN detected in prefix_neg")
        if torch.isnan(prefix_anomaly_subtype).any():
            raise ValueError("NaN detected in prefix_anomaly_subtype")

        if torch.isnan(suffix_pos).any():
            raise ValueError("NaN detected in suffix_pos")
        if torch.isnan(suffix_neg).any():
            raise ValueError("NaN detected in suffix_neg")
        if torch.isnan(suffix_anomaly_subtype).any():
            raise ValueError("NaN detected in suffix_anomaly_subtype")

        prompts_pos = torch.cat([prefix_pos, ctx_pos, suffix_pos], dim=2)
        prompts_neg = torch.cat([prefix_neg, ctx_neg, suffix_neg], dim=2)
        prompts_anomaly_subtype = torch.cat([prefix_anomaly_subtype, ctx_anomaly_subtype, suffix_anomaly_subtype], dim=2)

        if torch.isnan(prompts_pos).any():
            raise ValueError("NaN detected in prompts_pos after concatenation")
        if torch.isnan(prompts_neg).any():
            raise ValueError("NaN detected in prompts_neg after concatenation")
        if torch.isnan(prompts_anomaly_subtype).any():
            raise ValueError("NaN detected in prompts_anomaly_subtype after concatenation")

        _, _, l, d = prompts_pos.shape
        prompts_pos = prompts_pos.reshape(-1, l, d)
        _, _, l, d = prompts_neg.shape
        prompts_neg = prompts_neg.reshape(-1, l, d)
        _, _, l, d = prompts_anomaly_subtype.shape
        prompts_anomaly_subtype = prompts_anomaly_subtype.reshape(-1, l, d)

        prompts = torch.cat([prompts_pos, prompts_neg, prompts_anomaly_subtype], dim=0)

        _, l, d = self.tokenized_prompts_pos.shape
        tokenized_prompts_pos = self.tokenized_prompts_pos.reshape(-1, d)
        _, l, d = self.tokenized_prompts_neg.shape
        tokenized_prompts_neg = self.tokenized_prompts_neg.reshape(-1, d)
        _, l, d = self.tokenized_prompts_anomaly_subtype.shape
        tokenized_prompts_anomaly_subtype = self.tokenized_prompts_anomaly_subtype.reshape(-1, d)

        tokenized_prompts = torch.cat((tokenized_prompts_pos, tokenized_prompts_neg, tokenized_prompts_anomaly_subtype), dim=0)

        # Convert tokenized prompts back to text
        # decoded_prompts = [_tokenizer.decode(tokens.tolist()) for tokens in tokenized_prompts]

        # # Print the decoded prompts
        # for i, prompt in enumerate(decoded_prompts):
        #     print(f"Prompt {i}: {prompt}")
        if torch.isnan(prompts).any():
            raise ValueError(f"NaN detected in prompts during forward pass. prompts values: {prompts}")

        if torch.isnan(tokenized_prompts).any():
            raise ValueError(f"NaN detected in tokenized_prompts during forward pass. tokenized_prompts values: {tokenized_prompts}")

        return prompts, tokenized_prompts, self.compound_prompts_text
    
class Custom_AnomalyCLIP_PromptLearner2(nn.Module):
    def __init__(self, clip_model, design_details):
        super().__init__()
        classnames = ["object"]
        self.n_cls = len(classnames)
        self.n_ctx = design_details["Prompt_length"]
        
        # Initialize context vectors for normal, anomaly states, and anomaly subtypes
        n_ctx_pos = self.n_ctx
        n_ctx_neg = self.n_ctx
        n_ctx_anomaly_subtype = self.n_ctx
        
        self.text_encoder_n_ctx = design_details["learnable_text_embedding_length"]
        dtype = clip_model.transformer.get_cast_dtype()

        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.classnames = classnames

        self.state_normal_list = ["{}"]
        self.state_anomaly_list = ["damaged {}"]
        self.anomaly_subtype_list = ["anomaly type of"]

        normal_num = len(self.state_normal_list)
        anomaly_num = len(self.state_anomaly_list)
        anomaly_subtype_num = len(self.anomaly_subtype_list)

        self.normal_num = normal_num
        self.anomaly_num = anomaly_num
        self.anomaly_subtype_num = anomaly_subtype_num
        # Initialize context vectors for normal, anomaly states, and anomaly subtypes
        ctx_init_pos = ""
        ctx_init_neg = ""
        ctx_init_anomaly_subtype = ""
        # Initialize context vectors for normal, anomaly states, and anomaly subtypes
        # self.ctx_pos, self.prompt_prefix_pos = self.initialize_context(clip_model, "", n_ctx_pos, normal_num, ctx_dim, dtype)
        # self.ctx_neg, self.prompt_prefix_neg = self.initialize_context(clip_model, "", n_ctx_neg, anomaly_num, ctx_dim, dtype)
        # self.ctx_anomaly_subtype, self.prompt_prefix_anomaly_subtype = self.initialize_context(clip_model, "", n_ctx_anomaly_subtype, anomaly_subtype_num, ctx_dim, dtype)

        # Ensure ctx_pos and ctx_neg are registered as parameters
        # self.ctx_pos = nn.Parameter(self.ctx_pos)
        # self.ctx_neg = nn.Parameter(self.ctx_neg)
        # self.ctx_anomaly_subtype = nn.Parameter(self.ctx_anomaly_subtype)
        if ctx_init_pos and ctx_init_neg:
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            n_ctx_neg = len(ctx_init_neg.split(" "))
            prompt_pos = tokenize(ctx_init_pos)
            prompt_neg = tokenize(ctx_init_neg)
            with torch.no_grad():
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            ctx_vectors_pos_ = []
            ctx_vectors_neg_ = []
            for _ in range(self.n_cls):
                ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
            ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
            ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)
        else:
            print("Initializing class-specific contexts")
            ctx_vectors_pos = torch.empty(self.n_cls, normal_num, n_ctx_pos, ctx_dim, dtype=dtype)
            ctx_vectors_neg = torch.empty(self.n_cls, anomaly_num, n_ctx_neg, ctx_dim, dtype=dtype)
            ctx_vectors_anomaly_subtype = torch.empty(self.n_cls, anomaly_subtype_num, n_ctx_anomaly_subtype, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            nn.init.normal_(ctx_vectors_anomaly_subtype, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)
            prompt_prefix_anomaly_subtype = " ".join(["X"] * n_ctx_anomaly_subtype)
        
        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized
        self.ctx_anomaly_subtype = nn.Parameter(ctx_vectors_anomaly_subtype)  # to be optimized
        
        # Initialize compound prompts for depth and text
        self.compound_prompts_depth = design_details["learnable_text_embedding_depth"]
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            print("single_para", single_para.shape)
            nn.init.normal_(single_para, std=0.02)

        single_layer = nn.Linear(ctx_dim, 896)
        self.compound_prompt_projections = self._get_clones(single_layer, self.compound_prompts_depth - 1)

        # Prepare tokenized prompts
        classnames = [name.replace("_", " ") for name in self.classnames]

        prompts_pos = [prompt_prefix_pos + " " + template.format(name) + "." for template in self.state_normal_list for name in classnames]
        prompts_neg = [prompt_prefix_neg + " " + template.format(name) + "." for template in self.state_anomaly_list for name in classnames]
        prompts_anomaly_subtype = [prompt_prefix_anomaly_subtype + " " + template.format(name) + "." for template in self.anomaly_subtype_list for name in classnames]

        tokenized_prompts_pos = [tokenize(p) for p in prompts_pos]
        tokenized_prompts_neg = [tokenize(p) for p in prompts_neg]
        tokenized_prompts_anomaly_subtype = [tokenize(p) for p in prompts_anomaly_subtype]

        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)
        tokenized_prompts_anomaly_subtype = torch.cat(tokenized_prompts_anomaly_subtype)

        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            embedding_anomaly_subtype = clip_model.token_embedding(tokenized_prompts_anomaly_subtype).type(dtype)

            # Check for NaNs in the embeddings
            if torch.isnan(embedding_pos).any() or torch.isnan(embedding_neg).any() or torch.isnan(embedding_anomaly_subtype).any():
                raise ValueError("NaN detected in token embeddings")

            n, l, d = embedding_pos.shape
            embedding_pos = embedding_pos.reshape(self.normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_neg = embedding_neg.reshape(self.anomaly_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_anomaly_subtype = embedding_anomaly_subtype.reshape(self.anomaly_subtype_num, self.n_cls, l, d).permute(1, 0, 2, 3)

        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :])
        self.register_buffer("token_suffix_pos", embedding_pos[:, :, 1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:, :, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + n_ctx_neg:, :])
        self.register_buffer("token_prefix_anomaly_subtype", embedding_anomaly_subtype[:, :, :1, :])
        self.register_buffer("token_suffix_anomaly_subtype", embedding_anomaly_subtype[:, :, 1 + n_ctx_anomaly_subtype:, :])

        n, d = tokenized_prompts_pos.shape
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(self.normal_num, self.n_cls, d).permute(1, 0, 2)
        n, d = tokenized_prompts_neg.shape
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(self.anomaly_num, self.n_cls, d).permute(1, 0, 2)
        n, d = tokenized_prompts_anomaly_subtype.shape
        tokenized_prompts_anomaly_subtype = tokenized_prompts_anomaly_subtype.reshape(self.anomaly_subtype_num, self.n_cls, d).permute(1, 0, 2)

        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        self.n_ctx_anomaly_subtype = n_ctx_anomaly_subtype
        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)
        self.register_buffer("tokenized_prompts_anomaly_subtype", tokenized_prompts_anomaly_subtype)
        print("tokenized_prompts shape", self.tokenized_prompts_pos.shape, self.tokenized_prompts_neg.shape, self.tokenized_prompts_anomaly_subtype.shape)

    # def initialize_context(self, clip_model, ctx_init, n_ctx, num_templates, ctx_dim, dtype):
    #     if ctx_init:
    #         ctx_init = ctx_init.replace("_", " ")
    #         n_ctx = len(ctx_init.split(" "))
    #         prompt = tokenize(ctx_init)
    #         with torch.no_grad():
    #             embedding = clip_model.token_embedding(prompt).type(dtype)
    #         ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
    #         prompt_prefix = ctx_init
    #         ctx_vectors = torch.stack([deepcopy(ctx_vectors) for _ in range(self.n_cls)], dim=0)
    #     else:
    #         print("Initializing context vectors")
    #         ctx_vectors = torch.empty(self.n_cls, num_templates, n_ctx, ctx_dim, dtype=dtype)
    #         nn.init.normal_(ctx_vectors, std=0.02)
    #         prompt_prefix = " ".join(["X"] * n_ctx)

    #     # Check for NaNs in the initialized context vectors
    #     if torch.isnan(ctx_vectors).any():
    #         raise ValueError("NaN detected in context vectors during initialization")
        
    #     return nn.Parameter(ctx_vectors), prompt_prefix

    def _get_clones(self, module, N):
        return nn.ModuleList([deepcopy(module) for _ in range(N)])

    def forward(self, cls_id=None):
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg
        ctx_anomaly_subtype = self.ctx_anomaly_subtype

        prefix_pos = self.token_prefix_pos
        prefix_neg = self.token_prefix_neg
        prefix_anomaly_subtype = self.token_prefix_anomaly_subtype

        suffix_pos = self.token_suffix_pos
        suffix_neg = self.token_suffix_neg
        suffix_anomaly_subtype = self.token_suffix_anomaly_subtype

        # Debugging 信息
        # print("self.ctx_neg in def forward", self.ctx_neg)
        # print("ctx_neg after assignment from self.ctx_neg:", ctx_neg)

        if torch.isnan(self.ctx_neg).any():
            raise ValueError("NaN detected in self.ctx_neg before assignment")

        if torch.isnan(ctx_neg).any():
            raise ValueError("NaN detected in ctx_neg after assignment")

        if torch.isnan(ctx_pos).any():
            raise ValueError("NaN detected in ctx_pos")
        if torch.isnan(ctx_anomaly_subtype).any():
            raise ValueError("NaN detected in ctx_anomaly_subtype")

        if torch.isnan(prefix_pos).any():
            raise ValueError("NaN detected in prefix_pos")
        if torch.isnan(prefix_neg).any():
            raise ValueError("NaN detected in prefix_neg")
        if torch.isnan(prefix_anomaly_subtype).any():
            raise ValueError("NaN detected in prefix_anomaly_subtype")

        if torch.isnan(suffix_pos).any():
            raise ValueError("NaN detected in suffix_pos")
        if torch.isnan(suffix_neg).any():
            raise ValueError("NaN detected in suffix_neg")
        if torch.isnan(suffix_anomaly_subtype).any():
            raise ValueError("NaN detected in suffix_anomaly_subtype")

        prompts_pos = torch.cat([prefix_pos, ctx_pos, suffix_pos], dim=2)
        prompts_neg = torch.cat([prefix_neg, ctx_neg, suffix_neg], dim=2)
        prompts_anomaly_subtype = torch.cat([prefix_anomaly_subtype, ctx_anomaly_subtype, suffix_anomaly_subtype], dim=2)

        if torch.isnan(prompts_pos).any():
            raise ValueError("NaN detected in prompts_pos after concatenation")
        if torch.isnan(prompts_neg).any():
            raise ValueError("NaN detected in prompts_neg after concatenation")
        if torch.isnan(prompts_anomaly_subtype).any():
            raise ValueError("NaN detected in prompts_anomaly_subtype after concatenation")

        _, _, l, d = prompts_pos.shape
        prompts_pos = prompts_pos.reshape(-1, l, d)
        _, _, l, d = prompts_neg.shape
        prompts_neg = prompts_neg.reshape(-1, l, d)
        _, _, l, d = prompts_anomaly_subtype.shape
        prompts_anomaly_subtype = prompts_anomaly_subtype.reshape(-1, l, d)

        prompts = torch.cat([prompts_pos, prompts_neg, prompts_anomaly_subtype], dim=0)

        _, l, d = self.tokenized_prompts_pos.shape
        tokenized_prompts_pos = self.tokenized_prompts_pos.reshape(-1, d)
        _, l, d = self.tokenized_prompts_neg.shape
        tokenized_prompts_neg = self.tokenized_prompts_neg.reshape(-1, d)
        _, l, d = self.tokenized_prompts_anomaly_subtype.shape
        tokenized_prompts_anomaly_subtype = self.tokenized_prompts_anomaly_subtype.reshape(-1, d)

        tokenized_prompts = torch.cat((tokenized_prompts_pos, tokenized_prompts_neg, tokenized_prompts_anomaly_subtype), dim=0)

        # Convert tokenized prompts back to text
        # decoded_prompts = [_tokenizer.decode(tokens.tolist()) for tokens in tokenized_prompts]

        # # Print the decoded prompts
        # for i, prompt in enumerate(decoded_prompts):
        #     print(f"Prompt {i}: {prompt}")
        if torch.isnan(prompts).any():
            raise ValueError(f"NaN detected in prompts during forward pass. prompts values: {prompts}")

        if torch.isnan(tokenized_prompts).any():
            raise ValueError(f"NaN detected in tokenized_prompts during forward pass. tokenized_prompts values: {tokenized_prompts}")

        return prompts, tokenized_prompts, self.compound_prompts_text


import Meta_Inspector.AnomalyCLIP_lib
if __name__ == "__main__":
    n_ctx = 12
    depth = 9
    t_n_ctx= 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    AnomalyCLIP_parameters = {"Prompt_length": n_ctx, "learnable_text_embedding_depth": depth, "learnable_text_embedding_length": t_n_ctx}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
    model.eval()
    prompt_learner = Custom_AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    # prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
