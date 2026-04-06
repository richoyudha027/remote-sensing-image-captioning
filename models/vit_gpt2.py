import math
from PIL import Image


VIT_GPT2_WEIGHT_PATH = './weights/vit_gpt2_best_model.pt'


def build_vit_gpt2_model(tokenizer, device):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import ViTModel, GPT2LMHeadModel

    class SpatialPositionalEncoding(nn.Module):
        def __init__(self, embed_dim, grid_size=7):
            super().__init__()
            self.grid_size = grid_size
            self.row_embed = nn.Parameter(torch.randn(grid_size, embed_dim // 2) * 0.02)
            self.col_embed = nn.Parameter(torch.randn(grid_size, embed_dim // 2) * 0.02)

        def forward(self, batch_size, device):
            row_pos = self.row_embed.unsqueeze(1).repeat(1, self.grid_size, 1)
            col_pos = self.col_embed.unsqueeze(0).repeat(self.grid_size, 1, 1)
            spatial_pos = torch.cat([row_pos, col_pos], dim=-1)
            spatial_pos = spatial_pos.view(1, self.grid_size * self.grid_size, -1)
            return spatial_pos.expand(batch_size, -1, -1).to(device)

    class MultiScaleFeatureExtractor(nn.Module):
        def __init__(self, vit_dim, output_dim, layer_indices=[6, 9, 12]):
            super().__init__()
            self.layer_indices = layer_indices
            self.layer_weights = nn.Parameter(torch.ones(len(layer_indices)) / len(layer_indices))
            self.layer_norms = nn.ModuleList([nn.LayerNorm(vit_dim) for _ in layer_indices])
            self.fusion = nn.Sequential(
                nn.Linear(vit_dim, output_dim),
                nn.GELU(),
                nn.LayerNorm(output_dim)
            )

        def forward(self, hidden_states_list):
            weights = F.softmax(self.layer_weights, dim=0)
            fused = None
            for i, (hs, ln) in enumerate(zip(hidden_states_list, self.layer_norms)):
                normed = ln(hs)
                fused = weights[i] * normed if fused is None else fused + weights[i] * normed
            return self.fusion(fused)

    class SpatialAwareAttentionPooling(nn.Module):
        def __init__(self, embed_dim, num_heads=8, num_queries=49, dropout=0.1):
            super().__init__()
            self.num_queries = num_queries
            self.grid_size = int(math.sqrt(num_queries))
            self.query_tokens = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)
            self.spatial_pos = SpatialPositionalEncoding(embed_dim, self.grid_size)
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            self.self_attn  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout)
            )
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.norm3 = nn.LayerNorm(embed_dim)

        def forward(self, visual_features):
            batch_size = visual_features.shape[0]
            device = visual_features.device
            queries = self.query_tokens.expand(batch_size, -1, -1)
            queries = queries + self.spatial_pos(batch_size, device)
            attended, attn_w = self.cross_attn(queries, visual_features, visual_features)
            queries = self.norm1(queries + attended)
            self_att, _ = self.self_attn(queries, queries, queries)
            queries = self.norm2(queries + self_att)
            return self.norm3(queries + self.ffn(queries)), attn_w

    class ViTGPT2(nn.Module):
        def __init__(self, vit_model='google/vit-base-patch16-224', gpt2_model='gpt2',
                     num_image_tokens=49, freeze_vision=True, freeze_lm_initial_epochs=0,
                     dropout=0.15, multi_scale_layers=[6, 9, 12]):
            super().__init__()
            self.vit = ViTModel.from_pretrained(vit_model, output_hidden_states=True)
            self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model)
            self.vit_dim = self.vit.config.hidden_size
            self.gpt2_dim = self.gpt2.config.n_embd
            self.num_image_tokens = num_image_tokens
            self.multi_scale_layers = multi_scale_layers
            self.multi_scale_extractor = MultiScaleFeatureExtractor(
                self.vit_dim, self.vit_dim, multi_scale_layers
            )
            self.spatial_attention = SpatialAwareAttentionPooling(
                self.vit_dim, num_heads=12, num_queries=num_image_tokens, dropout=dropout
            )
            self.visual_projection = nn.Sequential(
                nn.Linear(self.vit_dim, self.gpt2_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(self.gpt2_dim, self.gpt2_dim), nn.LayerNorm(self.gpt2_dim)
            )
            self.freeze_vision = freeze_vision
            if freeze_vision:
                for param in self.vit.parameters():
                    param.requires_grad = False

        def encode_image(self, pixel_values):
            with torch.no_grad():
                vit_out = self.vit(pixel_values, output_hidden_states=True)
                selected = [vit_out.hidden_states[i].detach() for i in self.multi_scale_layers]
            multi_scale = self.multi_scale_extractor(selected)
            spatial, _ = self.spatial_attention(multi_scale)
            return self.visual_projection(spatial)

        def generate_caption(self, pixel_values, tokenizer, max_length=50, min_length=8,
                              num_beams=5, repetition_penalty=1.2, no_repeat_ngram_size=3,
                              device='cpu'):
            self.eval()
            with torch.no_grad():
                image_embeds = self.encode_image(pixel_values)
                batch_size = image_embeds.shape[0]
                bos_embeds = self.gpt2.transformer.wte(
                    torch.tensor([[tokenizer.bos_token_id]], device=device).expand(batch_size, -1)
                )
                inputs_embeds = torch.cat([image_embeds, bos_embeds], dim=1)
                attention_mask = torch.ones((batch_size, self.num_image_tokens + 1), device=device)
                outputs = self.gpt2.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_length,
                    min_new_tokens=min_length,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    early_stopping=True,
                    do_sample=False,
                    length_penalty=1.0
                )
                caption = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                if caption and caption[-1] not in '.!?':
                    caption += '.'
                return caption

    model = ViTGPT2(
        vit_model='google/vit-base-patch16-224',
        gpt2_model='gpt2',
        num_image_tokens=49,
        freeze_vision=True,
        dropout=0.1,
        multi_scale_layers=[6, 9, 12]
    )
    model.gpt2.resize_token_embeddings(len(tokenizer))
    return model


def load_vit_gpt2():
    print('[2/3] Loading ViT-GPT2.')
    import torch
    from transformers import GPT2Tokenizer, ViTImageProcessor

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({
        'bos_token': '<|startoftext|>',
        'eos_token': '<|endoftext|>',
        'pad_token': '<|pad|>'
    })

    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

    model = build_vit_gpt2_model(tokenizer, device)
    checkpoint = torch.load(VIT_GPT2_MODEL_PATH, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, feature_extractor, tokenizer, device


def predict_vit_gpt2(image_path, model, feature_extractor, tokenizer, device):
    image = Image.open(image_path).convert('RGB')
    pixel_values = feature_extractor(image, return_tensors='pt').pixel_values.to(device)
    return model.generate_caption(
        pixel_values=pixel_values,
        tokenizer=tokenizer,
        max_length=50,
        min_length=8,
        num_beams=5,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        device=device
    )