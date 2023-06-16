import argparse
import torch
import numpy as np
from SPMM_models import SPMM
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, WordpieceTokenizer
from dataset import SMILESDataset_pretrain
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical
from calc_property import calculate_property
from rdkit import Chem
import random
import pickle
from bisect import bisect_left


def BinarySearch(a, x):
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    else:
        return -1


def generate(model, image_embeds, text, stochastic=False, prop_att_mask=None):
    text_atts = torch.where(text == 0, 0, 1)
    if prop_att_mask is None:
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
    else:
        image_atts = prop_att_mask
    token_output = model.text_encoder(text,
                                      attention_mask=text_atts,
                                      encoder_hidden_states=image_embeds,
                                      encoder_attention_mask=image_atts,
                                      return_dict=True,
                                      is_decoder=True,
                                      return_logits=True,
                                      )[:, -1, :]  # batch*300
    if stochastic:
        p = torch.softmax(token_output, dim=-1)
        m = Categorical(p)
        token_output = m.sample()
    else:
        token_output = torch.argmax(token_output, dim=-1)
    return token_output.unsqueeze(1)  # batch*1


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device):
    # test
    print("PV-to-SMILES generation in deterministic manner...")
    model.eval()
    reference, candidate = [], []
    for (prop, text) in data_loader:
        prop = prop.to(device, non_blocking=True)
        property_feature = model.property_embed(prop.unsqueeze(2))
        property = torch.cat([model.property_cls.expand(property_feature.size(0), -1, -1), property_feature], dim=1)
        prop_embeds = model.property_encoder(inputs_embeds=property, return_dict=True).last_hidden_state

        text_input = torch.tensor([tokenizer.cls_token_id]).expand(prop.size(0), 1).to(device)
        end_count = torch.zeros_like(text_input).to(bool)
        for _ in range(100):
            output = generate(model, prop_embeds, text_input, stochastic=False)
            end_count = torch.logical_or(end_count, (output == tokenizer.sep_token_id))
            if end_count.all():
                break
            text_input = torch.cat([text_input, output], dim=-1)
        for i in range(text_input.size(0)):
            reference.append(text[i].replace('[CLS]', ''))
            sentence = text_input[i]
            if tokenizer.sep_token_id in sentence: sentence = sentence[:(sentence == tokenizer.sep_token_id).nonzero(as_tuple=True)[0][0].item()]
            cdd = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(sentence)).replace('[CLS]', '')
            candidate.append(cdd)
    print('Deterministic PV-to-SMILES generation done')
    return reference, candidate


@torch.no_grad()
def metric_eval(ref, cand):
    with open('./normalize.pkl', 'rb') as w:
        norm = pickle.load(w)
    property_mean, property_std = norm
    valids = []
    n_mse = []
    for i in range(len(ref)):
        try:
            prop_ref = calculate_property(ref[i])
            prop_cdd = calculate_property(cand[i])
            n_ref = (prop_ref - property_mean) / property_std
            n_cdd = (prop_cdd - property_mean) / property_std
            n_mse.append((n_ref - n_cdd) ** 2)
            valids.append(cand[i])
        except:
            continue
    if len(n_mse) != 0:
        n_mse = torch.stack(n_mse, dim=0)
        n_rmse = torch.sqrt(torch.mean(n_mse, dim=0))
    else:
        rmse, n_rmse = 0, 0
    print('mean of controlled properties\' normalized RMSE:', n_rmse.mean().item())

    lines = valids
    v = len(lines)
    print('validity:', v / len(cand))
    lines = [Chem.MolToSmiles(Chem.MolFromSmiles(l), isomericSmiles=False) for l in lines]
    lines = list(set(lines))
    u = len(lines)
    print('uniqueness:', u / v)

    with open('data/1_Pretrain/pretrain_20m.txt', 'r') as f:
        corpus = [l.strip() for l in f.readlines()]
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
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # ### Dataset ### #
    print("Creating dataset")
    dataset_test = SMILESDataset_pretrain(args.input_file)
    test_loader = DataLoader(dataset_test, batch_size=config['batch_size_test'], pin_memory=True, drop_last=False)

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

    print("=" * 50)
    r_test, c_test = evaluate(model, test_loader, tokenizer, device)
    metric_eval(r_test, c_test)
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./Pretrain/checkpoint_SPMM_20m.ckpt')
    parser.add_argument('--vocab_filename', default='./vocab_bpe_300.txt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--input_file', default='./data/2_PV2SMILES/pubchem_1k_unseen.txt', type=str)
    args = parser.parse_args()

    config = {
        'embed_dim': 256,
        'bert_config_text': './config_bert.json',
        'bert_config_property': './config_bert_property.json',
    }
    main(args, config)
