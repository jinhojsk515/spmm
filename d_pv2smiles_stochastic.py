import argparse
import torch
from SPMM_models import SPMM
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, WordpieceTokenizer
from calc_property import calculate_property
from rdkit import Chem
import random
import numpy as np
import pickle
import warnings
from d_pv2smiles_deterministic import BinarySearch, generate
warnings.filterwarnings(action='ignore')


@torch.no_grad()
def generate_with_property(model, property, n_sample, prop_mask, stochastic=True):
    device = model.device
    tokenizer = model.tokenizer
    # test
    model.eval()
    print("PV-to-SMILES generation in stochastic manner...")

    with open('./normalize.pkl', 'rb') as w:
        norm = pickle.load(w)
    property_mean, property_std = norm
    property = (property - property_mean) / property_std
    n_batch = 10
    prop = property.unsqueeze(0).repeat(n_sample // n_batch, 1)

    results = []
    prop = prop.to(device, non_blocking=True)
    property_feature = model.property_embed(prop.unsqueeze(2))

    property_unk = model.property_mask.expand(property_feature.size(0), property_feature.size(1), -1)
    mpm_mask_expand = prop_mask.unsqueeze(0).unsqueeze(2).repeat(property_unk.size(0), 1, property_unk.size(2)).to(device)
    property_masked = property_feature * (1 - mpm_mask_expand) + property_unk * mpm_mask_expand
    property = torch.cat([model.property_cls.expand(property_masked.size(0), -1, -1), property_masked], dim=1)
    prop_embeds = model.property_encoder(inputs_embeds=property, return_dict=True).last_hidden_state

    for _ in range(n_batch):
        text_input = torch.tensor([tokenizer.cls_token_id]).expand(prop.size(0), 1).to(device)
        end_count = torch.zeros_like(text_input).to(bool)
        for _ in range(100):
            output = generate(model, prop_embeds, text_input, stochastic=stochastic)
            end_count = torch.logical_or(end_count, (output == tokenizer.sep_token_id))
            if end_count.all():
                break
            text_input = torch.cat([text_input, output], dim=-1)
        for i in range(text_input.size(0)):
            sentence = text_input[i]
            if tokenizer.sep_token_id in sentence: sentence = sentence[:(sentence == tokenizer.sep_token_id).nonzero(as_tuple=True)[0][0].item()]
            cdd = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(sentence)).replace('[CLS]', '')
            results.append(cdd)
    return results


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

    with open('generated_molecules.txt', 'w') as w:
        for v in valids:    w.write(v + '\n')
    print('Generated molecules are saved in \'generated_molecules.txt\'')


def main(args, config):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = random.randint(0, 1000)
    print('seed:', seed)
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
    prop_mask = torch.zeros(53)         # 0 indicates no masking for that property
    prop_input = calculate_property('COc1cccc(NC(=O)CN(C)C(=O)COC(=O)c2cc(c3cccs3)nc3ccccc23)c1')

    '''condition for stochastic molecule generation of Fig.2-(b)'''
    # prop_mask = torch.ones(53)        # 1 indicates masking for that property
    # prop_mask[14] = 0
    # prop_input = torch.zeros(53)
    # prop_input[14] = 150

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
    samples = generate_with_property(model, prop_input, args.n_generate, prop_mask)
    metric_eval(prop_input, samples, prop_mask)
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./Pretrain/checkpoint_SPMM_20m.ckpt')
    parser.add_argument('--vocab_filename', default='./vocab_bpe_300.txt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n_generate', default=1000, type=int)
    args = parser.parse_args()

    config = {
        'embed_dim': 256,
        'bert_config_text': './config_bert.json',
        'bert_config_property': './config_bert_property.json',
    }
    main(args, config)
