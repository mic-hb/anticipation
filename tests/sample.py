import torch
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import GPT2LMHeadModel
from anticipation.vocabs.localmidi import vocab

def safe_midi(logits, idx):
    vocab_size = vocab['config']['size']
    if logits.size(-1) > vocab_size:
        logits[:, vocab_size:] = -float('inf')
    return logits

def generate(model, input_ids, tokens):
    model.eval()
    past_key_values = None
    input_ids = input_ids.unsqueeze(0)
    output_ids = input_ids.clone()
    for idx in tqdm(range(tokens)):
        with torch.no_grad():
            outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        logits = safe_midi(logits, idx)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).to(output_ids.device)
        output_ids = torch.cat([output_ids, next_token], dim=-1)
        input_ids = next_token
    return output_ids.cpu().numpy()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device)
    torch.manual_seed(args.seed)

    input_ids = torch.tensor([vocab['control_end']]).to(device)

    with open(args.output, 'w') as f:
        for i in range(args.sequences):
            output = generate(model, input_ids, args.tokens - len(input_ids))[0]
            f.write(' '.join(map(str, output)) + '\n')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-N', '--sequences', type=int, default=1)
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('-t', '--tokens', type=int, default=512)
    args = parser.parse_args()
    main(args)
