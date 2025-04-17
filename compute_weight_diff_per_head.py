import argparse
import torch
import wandb
from utils.factory import create_model_and_transforms, get_tokenizer

def get_args_parser():
    parser = argparse.ArgumentParser("Weight Difference", add_help=False)
    
    parser.add_argument(
        "--model",
        default="ViT-B-16",
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    parser.add_argument(
        "--pretrained",
        default="laion2b_s34b_b88k",
        type=str,
        help="Pretrained model to use",
    )
    parser.add_argument("--wandb_checkpoint", default=None, help="wandb checkpoint to load")
    parser.add_argument("--checkpoint_epoch", default=None, help="wandb project name")
    
    return parser
    
def main(args):
    model_pretrained, _, preprocess = create_model_and_transforms(args.model, pretrained=args.pretrained)
    
    model_finetuned, _, preprocess = create_model_and_transforms(args.model, pretrained=args.pretrained)
    if args.wandb_checkpoint:
        run = wandb.init()
        artifact = wandb.use_artifact(args.wandb_checkpoint, type='model')
        artifact_dir = artifact.download()
        checkpoint = torch.load(f"{artifact_dir}/{args.checkpoint_epoch}")
        model_finetuned.load_state_dict(checkpoint['model_state_dict'])
    
    if "ViT-B-16" in args.model:
        attn_weight_size = 768
        layers_we_care = [8, 9, 10, 11]
        heads_per_layer = range(12)
        single_head_input_size = 64
    
    pretrained_weights = {(i, j): {'W_q': 0, 'W_k': 0, 'W_v': 0} for i in layers_we_care for j in heads_per_layer}
    finetuned_weights = {(i, j): {'W_q': 0, 'W_k': 0, 'W_v': 0} for i in layers_we_care for j in heads_per_layer}
    QKV = {0: 'W_q', 1: 'W_k', 2: 'W_v'}
    
    for i in layers_we_care:
        for j in range(3): # 0: W_q, 1: W_k, 2: W_v
            for k in heads_per_layer:
                pretrained_weights[(i, k)][QKV[j]] = model_pretrained.state_dict()[f'visual.transformer.resblocks.{i}.attn.in_proj_weight'][j * attn_weight_size + k * single_head_input_size :j * attn_weight_size + (k + 1) * single_head_input_size, :]
                finetuned_weights[(i, k)][QKV[j]] = model_finetuned.state_dict()[f'visual.transformer.resblocks.{i}.attn.in_proj_weight'][j * attn_weight_size + k * single_head_input_size :j * attn_weight_size + (k + 1) * single_head_input_size, :]

    weight_diff = lambda layer, head, q_k_v: torch.norm(pretrained_weights[(layer, head)]['W_' + q_k_v] - finetuned_weights[(layer, head)]['W_' + q_k_v], p=2)
    diff_per_head = {(i, j): {'W_q': weight_diff(i, j, 'q'), 'W_k': weight_diff(i, j, 'k'), 'W_v': weight_diff(i, j, 'v')} for i in layers_we_care for j in heads_per_layer}
    diff_sorted_by_q = sorted(diff_per_head.items(), key=lambda x: x[1]['W_q'], reverse=True)
    diff_sorted_by_k = sorted(diff_per_head.items(), key=lambda x: x[1]['W_k'], reverse=True)
    diff_sorted_by_v = sorted(diff_per_head.items(), key=lambda x: x[1]['W_v'], reverse=True)
    
    print("Top 10 differences in W_q:")
    for i in range(10):
        print(f"Layer {diff_sorted_by_q[i][0][0]}, Head {diff_sorted_by_q[i][0][1]}: {diff_sorted_by_q[i][1]['W_q']}")
    print("Top 10 differences in W_k:")
    for i in range(10):
        print(f"Layer {diff_sorted_by_k[i][0][0]}, Head {diff_sorted_by_k[i][0][1]}: {diff_sorted_by_k[i][1]['W_k']}")
    print("Top 10 differences in W_v:")
    for i in range(10):
        print(f"Layer {diff_sorted_by_v[i][0][0]}, Head {diff_sorted_by_v[i][0][1]}: {diff_sorted_by_v[i][1]['W_v']}")
        
if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
        