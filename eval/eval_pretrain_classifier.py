import argparse
import json
import os
import sys

import shortuuid
import torch
from tqdm import tqdm

sys.path.append("./")
sys.path.append("./utils")

from model.language_model.llaga_llama_classifier import LlagaLlamaForSequenceClassification
from train.train_classifier import (
    DATASET_PATHS,
    load_pretrain_embedding_graph,
    load_pretrain_embedding_hop,
)
from utils import conversation as conversation_lib
from utils.constants import DEFAULT_GRAPH_PAD_ID, GRAPH_TOKEN_INDEX
from utils.utils import disable_torch_init, tokenizer_graph_token


def build_prompt(user_prompt, conv_mode):
    conv = conversation_lib.conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def eval_model(args):
    disable_torch_init()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        cache_dir=args.cache_dir,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    model = LlagaLlamaForSequenceClassification.from_pretrained(
        args.model_path,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).cuda()
    model.eval()

    data_dir = DATASET_PATHS[args.dataset]
    data = torch.load(os.path.join(data_dir, "processed_data.pt"), weights_only=False)
    if args.template == "ND":
        prompt_file = os.path.join(data_dir, f"sampled_{args.use_hop}_{args.sample_neighbor_size}_test.jsonl")
        pretrained_emb = load_pretrain_embedding_graph(data_dir, args.pretrained_embedding_type)
        structure_emb = torch.load(
            f"/localnvme/llaga/dataset/laplacian_{args.use_hop}_{args.sample_neighbor_size}.pt"
        )
    elif args.template == "HO":
        prompt_file = os.path.join(data_dir, "sampled_2_10_test.jsonl")
        pretrained_emb = load_pretrain_embedding_hop(data_dir, args.pretrained_embedding_type, args.use_hop)
        structure_emb = None
    else:
        raise ValueError(f"Unsupported template: {args.template}")

    with open(prompt_file, "r") as file:
        lines = file.readlines()

    if args.start >= 0:
        end = len(lines) if args.end < 0 else args.end
        lines = lines[args.start:end]
    elif args.end > 0:
        lines = lines[: args.end]

    output_dir = os.path.dirname(args.answers_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.answers_file, "w") as ans_file:
        for line in tqdm([json.loads(q) for q in lines], desc="Eval classifier"):
            prompt = build_prompt(line["conversations"][0]["value"], args.conv_mode)
            input_ids = tokenizer_graph_token(prompt, tokenizer, GRAPH_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

            if not isinstance(line["graph"][0], list):
                line["graph"] = [line["graph"]]

            if args.template == "ND":
                graph = torch.LongTensor(line["graph"])
                mask = graph != DEFAULT_GRAPH_PAD_ID
                masked_graph_emb = pretrained_emb[graph[mask]]
                sample_count, graph_len, hidden_dim = graph.shape[0], graph.shape[1], masked_graph_emb.shape[1]
                graph_emb = torch.zeros((sample_count, graph_len, hidden_dim))
                graph_emb[mask] = masked_graph_emb
                graph_emb = torch.cat([graph_emb, structure_emb.unsqueeze(0).expand(sample_count, -1, -1)], dim=-1)
            else:
                for g in range(len(line["graph"])):
                    center_id = line["graph"][g][0]
                    line["graph"][g] = [center_id] * (args.use_hop + 1)
                graph = torch.LongTensor(line["graph"])
                center_id = graph[:, 0]
                graph_emb = torch.stack([emb[center_id] for emb in pretrained_emb], dim=1)

            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_ids.ne(tokenizer.pad_token_id),
                    graph=graph.cuda(),
                    graph_emb=graph_emb.half().cuda(),
                )
                pred_id = int(outputs.logits.argmax(dim=-1).item())
                pooled = outputs.pooled_embeddings[0].float().cpu().tolist() if args.save_embeddings else None

            gt_id = int(data.y[line["id"]])
            ans_file.write(
                json.dumps(
                    {
                        "question_id": line["id"],
                        "text": data.label_texts[pred_id],
                        "pred_id": pred_id,
                        "gt_id": gt_id,
                        "gt": data.label_texts[gt_id],
                        "correct": int(pred_id == gt_id),
                        "answer_id": shortuuid.uuid(),
                        "embedding": pooled,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--pretrained_embedding_type", type=str, default="simteg")
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=10)
    parser.add_argument("--template", type=str, default="ND")
    parser.add_argument("--answers_file", type=str, required=True)
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--save_embeddings", action="store_true")
    args = parser.parse_args()
    eval_model(args)
