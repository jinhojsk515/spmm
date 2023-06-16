import argparse
import torch
import numpy as np
from transformers import BertTokenizer
from calc_property import calculate_property
import matplotlib.pyplot as plt
from rdkit import Chem
import pickle
from SPMM_models import SPMM


def get_attention_map(model, property_original, text_input_ids, text_attention_mask):
    property1 = model.property_embed(property_original.unsqueeze(2))
    properties = torch.cat([model.property_cls.expand(property_original.size(0), -1, -1), property1], dim=1)
    prop_embeds = model.property_encoder(inputs_embeds=properties, return_dict=True, is_decoder=False).last_hidden_state
    prop_atts = torch.ones(prop_embeds.size()[:-1], dtype=torch.long).to(properties.device)

    text_output = model.text_encoder.bert(text_input_ids, attention_mask=text_attention_mask, return_dict=True, mode='text', output_attentions=True)
    text_embeds = text_output.last_hidden_state
    text_attentions = text_output.attentions

    output = model.text_encoder.bert(encoder_embeds=prop_embeds,
                                     attention_mask=prop_atts,
                                     encoder_hidden_states=text_embeds,
                                     encoder_attention_mask=text_attention_mask,
                                     return_dict=True,
                                     mode='fusion',
                                     output_attentions=True,
                                     is_decoder=False
                                     )
    prop_attentions = output.attentions
    cross_attentions = output.cross_attentions
    return prop_attentions, cross_attentions


def main(args, config):
    device = torch.device(args.device)
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=False, do_basic_tokenize=False)

    # === Model === #
    print("Creating model")
    model = SPMM(config=config, tokenizer=tokenizer, no_train=True)
    model.eval()
    print('#parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.checkpoint:
        print('LOADING PRETRAINED MODEL..')
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['state_dict']

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        # print(msg)
    model = model.to(device)

    print("=" * 50)
    # preprocess input
    print("input SMILES: ", args.input_smiles)
    prop_smiles = args.input_smiles
    prop_smiles = '[CLS]' + Chem.MolToSmiles(Chem.MolFromSmiles(prop_smiles), isomericSmiles=False, canonical=True)
    prop_input = calculate_property(args.input_smiles).unsqueeze(0)
    with open('./normalize.pkl', 'rb') as w:
        property_mean, property_std = pickle.load(w)
    prop_input = (prop_input - property_mean) / property_std
    prop_input = prop_input.to(device)
    text_input = tokenizer([prop_smiles], padding='longest', truncation=True, max_length=100, return_tensors="pt").to(device)

    # get attention map
    with torch.no_grad():
        attentions, cross_attentions = get_attention_map(model, prop_input, text_input.input_ids[:, 1:], text_input.attention_mask[:, 1:])

    # show attention map figure
    plt.figure(figsize=(8, 6))

    presenting_properties = [42, 41, 14, 50, 40, 45, 51, 20, 30, 31, 34, 52]
    tokens = tokenizer.convert_ids_to_tokens(text_input.input_ids[0])[2:-1]     # remove [CLS] and [SEP]
    tokens = [t[2:] for t in tokens]
    plt.matshow(cross_attentions[-1][0, :, :, :].cpu().mean(0)[presenting_properties, 1:-1].clamp(max=1.5 / len(tokens)), cmap='BuGn', fignum=1)
    ax = plt.gca()
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticks(np.arange(12))
    with open('./property_name.txt', 'r') as f:
        names = [l.strip() for l in f.readlines()[:53]]
        names = [names[i] for i in presenting_properties]
    ax.set_yticklabels(names)
    # plt.savefig('av.png', dpi=300)
    # print("attention map figure saved as av.png")
    plt.show()

    print("attention visualization finished")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./Pretrain/checkpoint_SPMM_20m.ckpt')
    parser.add_argument('--vocab_file', default='./vocab_bpe_300.txt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--input_smiles', default='CCN(C)CCC(O)C(c1ccccc1)c1ccccc1', type=str)
    args = parser.parse_args()

    cls_config = {
        'embed_dim': 256,
        'bert_config_text': './config_bert.json',
        'bert_config_property': './config_bert_property.json',
    }
    main(args, cls_config)
