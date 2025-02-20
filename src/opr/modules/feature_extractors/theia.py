import transformers
from transformers import AutoModel
import torch
import torch.nn as nn

# model = AutoModel.from_pretrained("theaiinstitute/theia-base-patch16-224-cdiv", trust_remote_code=True)
# fake_input = torch.zeros((1, 224, 224, 3), dtype=torch.uint8)

# theia_feature = model.forward_feature(fake_input)
# Theia / intermediate feature, mainly used for robot learning.
# To change different feature reduction methods, pass `feature_reduction_method` argument in AutoModel.from_pretrained() method

# predicted_features = model(fake_input)

# print(f"keys: {predicted_features.keys()}")
# predicted_features is dict[str, torch.Tensor] where each kv pair is target model name and predicted feature
# they are predicted features that tries to match teacher model features.

# print(f"theia_feature.shape: {theia_feature.shape}")
# print(f"predicted_features['facebook/dinov2-large'].shape: {predicted_features['facebook/dinov2-large'].shape}")
# print(f"predicted_features['google/vit-huge-patch14-224-in21k'].shape: {predicted_features['google/vit-huge-patch14-224-in21k'].shape}")
# print(f"predicted_features['openai/clip-vit-large-patch14'].shape: {predicted_features['openai/clip-vit-large-patch14'].shape}")

target_model_names = [
 "google/vit-huge-patch14-224-in21k",
 "facebook/dinov2-large",
 "openai/clip-vit-large-patch14"]

class TheiaFeatureExtractor(nn.Module):
    def __init__(self,
                 feat_type: str="theia", 
                 device: str="cpu"):
        super().__init__()

        self.feat_type = "theia"
        self.model = AutoModel.from_pretrained("theaiinstitute/theia-base-patch16-224-cdiv", 
                                               trust_remote_code=True,
                                               )

        print(self.model.backbone)

    def forward(self, img: torch.Tensor) -> torch.Tensor:

        # print(f"img")
        # print(f"img.shape: {img.shape}")

        # print(f"self.model.num_reg_tokens: {self.model.num_reg_tokens}")
        # print(f"self.model.translator: {self.model.translator}")
        # print(f"self.model.forward_neck: {self.model.forward_neck}")
        # print(f"self.model.no_cls: {self.model.no_cls}")

        # features = self.translator(x, target_model_names, backbone_no_cls=self.no_cls)
        with torch.no_grad():

            if self.feat_type == "theia":
                # desc = self.model.forward_feature(img)
                # return desc
                
                # (86, 768)
                # desc = self.model.backbone(img)[:,0,:].unsqueeze(1) 
                # (2745, 768)
                # desc = self.model.backbone(img, interpolate_pos_encoding=True, do_rescale=False)

                # desc = self.model.forward(img, interpolate_pos_encoding=True, do_rescale=False)

                # self.model.neck = nn.Identity()
                # self.model.forward_neck=True
                # print(self.model)
                # desc = self.model.forward(img, interpolate_pos_encoding=True, do_rescale=False)

                # desc = self.model.forward_feature(img)

                # self.model.translator = None

                # print(f"self.preprocessor: {self.preprocessor}")
                # print(f"self.model.backbone.processor: {self.model.backbone.processor}")

                # self.model.backbone.processor.

                input = self.model.backbone.processor(
                    img, return_tensors="pt", do_rescale=True, do_resize=False
                ).to(self.model.device)
                y = self.model.backbone.model(**input, interpolate_pos_encoding=True)
                # y : [batch_size, n_tokens, token_dim]
                y = y.last_hidden_state
                y = y[:,0,:].unsqueeze(1)

                # print(f"y.shape: {y.shape}")

                return y

                # feature = self.backbone(x, **kwargs)
                # [B, 1+H*W+N, C] if including both CLS and register tokens.
                # [B, 1+H*W, C] for standard model (N=0).
                # [B, H*W, C] for model without CLS.
                return handle_feature_output(feature, num_discard_tokens=self.num_reg_tokens)

                input = self.model.backbone.processor(
                    img, return_tensors="pt", do_resize=False, do_rescale=False, do_normalize=True
                ).to("cuda")

                y = self.model.backbone.model(**input, interpolate_pos_encoding=True)
                desc = y.last_hidden_state

                desc = desc[:,1:,:]
                # [32, 401, 768]

                # print(f"y: {y}")
                # return y.last_hidden_state

                

                # print(f"desc.shape: {desc.shape}")

                # desc = self.model.translator(desc, target_model_names, backbone_no_cls=self.model.no_cls)

                # desc = self.model.backbone.model.forward(img, interpolate_pos_encoding=True, do_rescale=False)

                # print(f"self.model.target_feature_sizes: {self.model.target_feature_sizes}")
                

                # print(f"desc.shape: {desc.shape}")
                return desc
            
            elif self.feat_type=="facebook/dinov2-large" or \
                 self.feat_type=="google/vit-huge-patch14-224-in21k" or \
                 self.feat_type=="openai/clip-vit-large-patch14":
                return self.model()["self.feat_type"]
            else:
                raise ValueError("unknown feature type")
            

# extractor = TheiaFeatureExtractor().to("cuda")
# fake_input = torch.zeros((1, 320, 320, 3), dtype=torch.uint8).to("cuda")
# print(f"extractor(fake_input): {extractor(fake_input).shape}")

