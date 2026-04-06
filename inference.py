import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import json
import pickle
import math
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


MODELS_DIR = './models'

VIT_BILSTM_MODEL_PATH     = os.path.join(MODELS_DIR, 'vit_bilstm_best_model.h5')
VIT_BILSTM_TOKENIZER_PATH = os.path.join(MODELS_DIR, 'vit_bilstm_tokenizer.json')
VIT_BILSTM_CONFIG_PATH    = os.path.join(MODELS_DIR, 'vit_bilstm_config.json')
VIT_GPT2_MODEL_PATH       = os.path.join(MODELS_DIR, 'vit_gpt2_best_model.pt')
REMOTECLIP_MODEL_PATH     = os.path.join(MODELS_DIR, 'remoteclip_gpt2_best_model.pt')

# -----------------------------------------------------
#                      ViT-BiLSTM
# -----------------------------------------------------

def load_vit_bilstm():
    print('[1/3] Loading ViT-BiLSTM.')
    import tensorflow as tf
    import tensorflow_hub as hub
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.layers import (Input, Dropout, Dense, Embedding, LSTM,
                                          Reshape, concatenate, Bidirectional,
                                          LayerNormalization, Flatten, add)
    from tensorflow.keras.models import Model

    with open(VIT_BILSTM_TOKENIZER_PATH, 'rb') as f:
        tokenizer = tokenizer_from_json(f.read())
    with open(VIT_BILSTM_CONFIG_PATH, 'r') as f:
        config = json.load(f)
    max_length = config['max_length']
    vocab_size = config['vocab_size']

    # ViT Feature Extractor
    vit_url = 'https://tfhub.dev/sayakpaul/vit_b16_fe/1'
    class ViTFeatureExtractor(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.vit = hub.KerasLayer(vit_url, trainable=False)
        def call(self, images):
            return self.vit((images * 2.0) - 1.0)

    feature_extractor = ViTFeatureExtractor()

    # Arsitektur Model
    input1 = Input(shape=(768,))
    input2 = Input(shape=(max_length,))

    img_features = Dense(512, activation='relu')(input1)
    img_features = Dropout(0.3)(img_features)
    img_features = Dense(512, activation='relu')(img_features)
    img_features_reshaped = Reshape((1, 512))(img_features)

    sentence_features = Embedding(vocab_size, 512, mask_zero=False)(input2)
    merged = concatenate([img_features_reshaped, sentence_features], axis=1)

    bi_lstm_out = Bidirectional(LSTM(512, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(merged)
    bi_lstm_out = LayerNormalization()(bi_lstm_out)
    bi_lstm_out2 = Bidirectional(LSTM(512, dropout=0.2, recurrent_dropout=0.2))(bi_lstm_out)
    bi_lstm_out2 = LayerNormalization()(bi_lstm_out2)

    X = Dense(512, activation='relu')(bi_lstm_out2)
    X = Dropout(0.4)(X)

    img_features_flat = Flatten()(img_features_reshaped)
    X = add([X, img_features_flat])

    X = Dense(256, activation='relu')(X)
    X = Dropout(0.5)(X)
    output = Dense(vocab_size, activation='softmax')(X)

    caption_model = Model(inputs=[input1, input2], outputs=output)
    caption_model.load_weights(VIT_BILSTM_MODEL_PATH)

    return caption_model, feature_extractor, tokenizer, max_length, pad_sequences

# -----------------------------------------------------
#                       ViT-GPT2
# -----------------------------------------------------

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

# -----------------------------------------------------
#                   RemoteCLIP-GPT2
# -----------------------------------------------------

def build_remoteclip_model(clip_model, gpt2_model, tokenizer):
    import torch
    import torch.nn as nn

    class SimpleVisionLanguageProjector(nn.Module):
        def __init__(self, clip_dim=512, gpt2_dim=768, num_tokens=8):
            super().__init__()
            self.num_tokens = num_tokens
            self.projection = nn.Sequential(
                nn.Linear(clip_dim, gpt2_dim * 2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(gpt2_dim * 2, gpt2_dim * num_tokens), nn.Dropout(0.1)
            )
            for layer in self.projection:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)

        def forward(self, clip_features):
            return self.projection(clip_features).view(clip_features.shape[0], self.num_tokens, -1)

    class RemoteCLIPGPT2(nn.Module):
        def __init__(self, clip_model, gpt2_model, clip_dim=512, gpt2_dim=768, num_visual_tokens=8):
            super().__init__()
            self.clip_model = clip_model
            self.gpt2_model = gpt2_model
            self.projector = SimpleVisionLanguageProjector(clip_dim, gpt2_dim, num_visual_tokens)
            self.num_visual_tokens = num_visual_tokens
            for param in self.clip_model.parameters():
                param.requires_grad = False

        def generate_caption(self, image, tokenizer, max_length=50, num_beams=5,
                              repetition_penalty=1.2, no_repeat_ngram_size=3):
            self.eval()
            with torch.no_grad():
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                clip_features = self.clip_model(image)
                visual_embeds = self.projector(clip_features)
                batch_size = visual_embeds.shape[0]
                bos_embeds = self.gpt2_model.transformer.wte(
                    torch.tensor([[tokenizer.bos_token_id]], device=image.device).expand(batch_size, -1)
                )
                inputs_embeds = torch.cat([visual_embeds, bos_embeds], dim=1)
                attention_mask = torch.ones((batch_size, self.num_visual_tokens + 1), device=image.device)
                outputs = self.gpt2_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_length,
                    min_new_tokens=5,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    early_stopping=True,
                    do_sample=False
                )
                caption = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                if caption and caption[-1] not in '.!?':
                    caption += '.'
                return caption

    return RemoteCLIPGPT2(clip_model, gpt2_model, clip_dim=512, gpt2_dim=768, num_visual_tokens=8)

def load_remoteclip_gpt2():
    print('[3/3] Loading RemoteCLIP-GPT2.')
    import torch
    import open_clip
    from huggingface_hub import hf_hub_download
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({
        'bos_token': '<|startoftext|>',
        'eos_token': '<|endoftext|>',
        'pad_token': '<|pad|>'
    })

    clip_ckpt_path = hf_hub_download(repo_id='chendelong/RemoteCLIP', filename='RemoteCLIP-ViT-B-32.pt')
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None)
    clip_checkpoint = torch.load(clip_ckpt_path, map_location='cpu')
    clip_model.load_state_dict(clip_checkpoint)
    clip_model = clip_model.visual.to(device)
    clip_model.eval()

    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_model.resize_token_embeddings(len(tokenizer))

    model = build_remoteclip_model(clip_model, gpt2_model, tokenizer)
    checkpoint = torch.load(REMOTECLIP_MODEL_PATH, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()

    return model, preprocess, tokenizer, device


# -----------------------------------------------------
#                  INFERENCE FUNCTIONS
# -----------------------------------------------------

def predict_vit_bilstm(image_path, caption_model, feature_extractor, tokenizer, max_length, pad_sequences):
    import tensorflow as tf

    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32) / 255.0
    img = (img * 2.0) - 1.0
    img = tf.expand_dims(img, 0)
    img_feature = feature_extractor.predict(img, verbose=0)

    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([img_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = None
        for w, idx in tokenizer.word_index.items():
            if idx == yhat:
                word = w
                break
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word

    final_caption = in_text.replace('startseq', '').strip()
    if final_caption and final_caption[-1] not in '.!?':
        final_caption += '.'
    return final_caption


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


def predict_remoteclip(image_path, model, preprocess, tokenizer, device):
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    return model.generate_caption(
        image_tensor,
        tokenizer,
        max_length=50,
        num_beams=5,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )

# -----------------------------------------------------
#                       Display
# -----------------------------------------------------

def wrap_text(text, max_chars=55):
    words = text.split()
    lines, line = [], []
    for word in words:
        line.append(word)
        if len(' '.join(line)) > max_chars:
            lines.append(' '.join(line[:-1]))
            line = [word]
    if line:
        lines.append(' '.join(line))
    return lines


def display_results(image_path, captions: dict):
    image = Image.open(image_path).convert('RGB')

    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor("#ECECEC")

    ax_img = fig.add_axes([0.02, 0.08, 0.43, 0.84])
    ax_img.imshow(image)
    ax_img.axis('off')
    ax_img.set_title(os.path.basename(image_path), color='black', fontsize=11, pad=8)

    ax_text = fig.add_axes([0.48, 0.05, 0.50, 0.90])
    ax_text.set_facecolor('#ECECEC')
    ax_text.axis('off')

    y = 0.95
    ax_text.text(0.03, y, 'Generated Captions', color='black',
                 fontsize=13, fontweight='bold', transform=ax_text.transAxes)
    y -= 0.10

    for model_name, caption in captions.items():
        ax_text.text(0.03, y, f'[{model_name}]', color='black',
                     fontsize=10, fontweight='bold', transform=ax_text.transAxes)
        y -= 0.06

        for line in wrap_text(caption):
            ax_text.text(0.03, y, line, color='black',
                         fontsize=10, transform=ax_text.transAxes)
            y -= 0.06

        y -= 0.03

    plt.suptitle('Remote Sensing Image Captioning — Comparison',
                 color='black', fontsize=14, fontweight='bold', y=0.99)
    plt.show()


# -----------------------------------------------------
#                        Main
# -----------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='RSIC Inference — 3 Model Comparison')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--no-display', action='store_true', help='Only print captions without displaying the image')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f'Error: File not found — {args.image}')
        sys.exit(1)

    print(f'\nImage  : {args.image}')
    print('\nLoading models.')

    bilstm_assets     = load_vit_bilstm()
    vit_gpt2_assets   = load_vit_gpt2()
    remoteclip_assets = load_remoteclip_gpt2()

    print('\nRunning inference.')

    cap_bilstm     = predict_vit_bilstm(args.image, *bilstm_assets)
    cap_vitgpt2    = predict_vit_gpt2(args.image, *vit_gpt2_assets)
    cap_remoteclip = predict_remoteclip(args.image, *remoteclip_assets)

    captions = {
        'ViT-BiLSTM':      cap_bilstm,
        'ViT-GPT2':        cap_vitgpt2,
        'RemoteCLIP-GPT2': cap_remoteclip,
    }

    print()
    for model_name, caption in captions.items():
        print(f'[{model_name}] {caption}')

    if not args.no_display:
        display_results(args.image, captions)


if __name__ == '__main__':
    main()