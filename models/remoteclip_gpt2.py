from PIL import Image


REMOTECLIP_WEIGHT_PATH = './weights/remoteclip_gpt2_best_model.pt'


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