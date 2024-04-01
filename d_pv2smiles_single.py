import argparse
import torch
from SPMM_models import SPMM
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, WordpieceTokenizer
from calc_property import calculate_property
from torch.distributions.categorical import Categorical
from rdkit import Chem
import random
import numpy as np
import pickle
import warnings
from tqdm import tqdm
from bisect import bisect_left
warnings.filterwarnings(action='ignore')


def BinarySearch(a, x):
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    else:
        return -1


def generate(model, image_embeds, text, stochastic=True, prop_att_mask=None, k=None):
    text_atts = torch.where(text == 0, 0, 1)
    if prop_att_mask is None:   prop_att_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
    token_output = model.text_encoder(text,
                                      attention_mask=text_atts,
                                      encoder_hidden_states=image_embeds,
                                      encoder_attention_mask=prop_att_mask,
                                      return_dict=True,
                                      is_decoder=True,
                                      return_logits=True,
                                      )[:, -1, :]  # batch*300
    if k:
        p = torch.softmax(token_output, dim=-1)
        if stochastic:
            output = torch.multinomial(p, num_samples=k, replacement=False)
            return torch.log(torch.stack([p[i][output[i]] for i in range(output.size(0))])), output
        else:
            output = torch.topk(p, k=k, dim=-1)  # batch*k
            return torch.log(output.values), output.indices
    if stochastic:
        p = torch.softmax(token_output, dim=-1)
        m = Categorical(p)
        token_output = m.sample()
    else:
        token_output = torch.argmax(token_output, dim=-1)
    return token_output.unsqueeze(1)  # batch*1


@torch.no_grad()
def generate_with_property(model, properties, n_sample, prop_mask, k=2, stochastic=True):
    device = model.device
    tokenizer = model.tokenizer
    # test
    model.eval()
    print(f"PV-to-SMILES generation in {'stochastic' if stochastic else 'deterministic'} manner with k={k}...")

    with open('./normalize.pkl', 'rb') as w:
        norm = pickle.load(w)
    property_mean, property_std = norm
    properties = (properties - property_mean) / property_std
    prop = properties.unsqueeze(0).repeat(1, 1)
    prop = prop.to(device, non_blocking=True)

    property1 = model.property_embed(prop.unsqueeze(2))  # batch*12*feature

    property_unk = model.property_mask.expand(property1.size(0), property1.size(1), -1)
    mpm_mask_expand = prop_mask.unsqueeze(0).unsqueeze(2).repeat(property_unk.size(0), 1, property_unk.size(2)).to(device)
    property_masked = property1 * (1 - mpm_mask_expand) + property_unk * mpm_mask_expand

    properties = torch.cat([model.property_cls.expand(property_masked.size(0), -1, -1), property_masked], dim=1)
    prop_embeds = model.property_encoder(inputs_embeds=properties, return_dict=True).last_hidden_state  # batch*len(=patch**2+1)*feature

    candidate = []
    for _ in tqdm(range(n_sample)):
        product_input = torch.tensor([tokenizer.cls_token_id]).expand(1, 1).to(device)
        values, indices = generate(model, prop_embeds, product_input, stochastic=stochastic, k=k)
        # print(values, indices, values.size(), indices.size())
        product_input = torch.cat([torch.tensor([tokenizer.cls_token_id]).expand(k, 1).to(device), indices.squeeze(0).unsqueeze(-1)], dim=-1)
        current_p = values.squeeze(0)
        final_output = []
        for _ in range(100):
            values, indices = generate(model, prop_embeds, product_input, stochastic=stochastic, k=k)
            k2_p = current_p[:, None] + values
            product_input_k2 = torch.cat([product_input.unsqueeze(1).repeat(1, k, 1), indices.unsqueeze(-1)], dim=-1)
            if tokenizer.sep_token_id in indices:
                ends = (indices == tokenizer.sep_token_id).nonzero(as_tuple=False)
                for e in ends:
                    p = k2_p[e[0], e[1]].cpu().item()
                    final_output.append((p, product_input_k2[e[0], e[1]]))
                    k2_p[e[0], e[1]] = -1e5
                if len(final_output) >= k ** 2:
                    break
            current_p, i = torch.topk(k2_p.flatten(), k)
            next_indices = torch.from_numpy(np.array(np.unravel_index(i.cpu().numpy(), k2_p.shape))).T
            product_input = torch.stack([product_input_k2[i[0], i[1]] for i in next_indices], dim=0)

        candidate_k = []
        final_output = sorted(final_output, key=lambda x: x[0], reverse=True)[:k]
        for p, sentence in final_output:
            cdd = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(sentence[:-1])).replace('[CLS]', '')
            candidate_k.append(cdd)
        if not stochastic:
            candidate.append(candidate_k[0])
        else:
            candidate.append(random.choice(candidate_k))
    return candidate


@torch.no_grad()
def metric_eval(prop_input, cand, mask):
    with open('./normalize.pkl', 'rb') as w:
        norm = pickle.load(w)

    random.shuffle(cand)
    mse = []
    valids = []
    prop_cdds = []
    for i in range(len(cand)):
        try:
            prop_cdd = calculate_property(cand[i])
            n_ref = (prop_input - norm[0]) / norm[1]
            n_cdd = (prop_cdd - norm[0]) / norm[1]
            mse.append((n_ref - n_cdd) ** 2)
            prop_cdds.append(prop_cdd)
            valids.append(cand[i])
        except:
            continue
    mse = torch.stack(mse, dim=0)
    rmse = torch.sqrt(torch.mean(mse, dim=0))
    print('mean of controlled properties\' normalized RMSE:', rmse[(1 - mask).long().bool()].mean().item())
    valids = [Chem.MolToSmiles(Chem.MolFromSmiles(l), isomericSmiles=False, canonical=True) for l in valids]

    lines = valids
    v = len(lines)
    print('validity:', v / len(cand))

    lines = [Chem.MolToSmiles(Chem.MolFromSmiles(l), isomericSmiles=False) for l in lines]
    lines = list(set(lines))
    u = len(lines)
    print('uniqueness:', u / v)

    '''
    corpus = []
    with open('data/1_Pretrain/pretrain_20m.txt', 'r') as f:
        for _ in range(20000000):
            corpus.append(f.readline().strip())
    corpus.sort()
    count = 0
    for l in lines:
        if BinarySearch(corpus, l) < 0:
            count += 1
    print('novelty:', count / u)
    '''

    with open('generated_molecules.txt', 'w') as w:
        for v in valids:    w.write(v + '\n')
    print('Generated molecules are saved in \'generated_molecules.txt\'')


def main(args, config):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = random.randint(0, 1000)
    print('seed:', seed, args.stochastic)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)

    # === Model === #
    print("Creating model")
    model = SPMM(config=config, tokenizer=tokenizer, no_train=True)

    if args.checkpoint:
        print('LOADING PRETRAINED MODEL..')
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['state_dict']

        for key in list(state_dict.keys()):
            if 'word_embeddings' in key and 'property_encoder' in key:
                del state_dict[key]
            if 'queue' in key:
                del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)
    model = model.to(device)

    '''condition for stochastic molecule generation of Fig.2-(a)'''
    # prop_mask = torch.zeros(53)  # 0 indicates no masking for that property
    # prop_input = calculate_property('COc1cccc(NC(=O)CN(C)C(=O)COC(=O)c2cc(c3cccs3)nc3ccccc23)c1')

    '''condition for stochastic molecule generation of Fig.2-(b)'''
    prop_mask = torch.ones(53)        # 1 indicates masking for that property
    prop_mask[14] = 0
    prop_input = torch.zeros(53)
    prop_input[14] = 150

    '''condition for stochastic molecule generation of Fig.2-(c)'''
    # prop_mask = torch.ones(53)
    # prop_mask[[50, 40, 51, 52]] = 0
    # prop_input = torch.zeros(53)
    # prop_input[50] = 2
    # prop_input[40] = 1
    # prop_input[51] = 30
    # prop_input[52] = 0.8

    '''condition for stochastic molecule generation of Fig.2-(d)'''
    # prop_mask = torch.ones(53)
    # prop_input = torch.zeros(53)

    print("=" * 50)
    samples = generate_with_property(model, prop_input, args.n_generate, prop_mask, stochastic=args.stochastic, k=args.k)
    metric_eval(prop_input, samples, prop_mask)
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./Pretrain/checkpoint_SPMM.ckpt')
    parser.add_argument('--vocab_filename', default='./vocab_bpe_300.txt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n_generate', default=1000, type=int)
    parser.add_argument('--k', default=2, type=int)
    parser.add_argument('--stochastic', default=True, type=bool)
    arg = parser.parse_args()

    configs = {
        'embed_dim': 256,
        'bert_config_text': './config_bert.json',
        'bert_config_property': './config_bert_property.json',
    }
    main(arg, configs)
