import argparse
import torch
import numpy as np
from SPMM_models import SPMM
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, WordpieceTokenizer
from dataset import SMILESDataset_pretrain
from torch.utils.data import DataLoader
import random
import pickle
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def generate(model, prop_input, text_embeds, text_atts):
    prop_embeds = model.property_encoder(inputs_embeds=prop_input, return_dict=True).last_hidden_state
    prob_atts = torch.ones(prop_input.size()[:-1], dtype=torch.long).to(prop_input.device)
    token_output = model.text_encoder.bert(encoder_embeds=prop_embeds,
                                           attention_mask=prob_atts,
                                           encoder_hidden_states=text_embeds,
                                           encoder_attention_mask=text_atts,
                                           return_dict=True,
                                           is_decoder=True,
                                           mode='fusion',
                                           ).last_hidden_state
    pred = model.property_mtr_head(token_output).squeeze(-1)[:, -1]
    return pred.unsqueeze(1)


@torch.no_grad()
def pv_generate(model, data_loader):
    # test
    with open('./normalize.pkl', 'rb') as w:
        mean, std = pickle.load(w)
    device = model.device
    tokenizer = model.tokenizer
    model.eval()
    print("SMILES-to-PV generation...")
    # convert list of string to dataloader
    if isinstance(data_loader, list):
        gather = []
        text_input = tokenizer(data_loader, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(device)
        text_embeds = model.text_encoder.bert(text_input.input_ids[:, 1:], attention_mask=text_input.attention_mask[:, 1:],
                                              return_dict=True, mode='text').last_hidden_state
        prop_input = model.property_cls.expand(len(data_loader), -1, -1)
        prediction = []
        for _ in range(53):
            output = generate(model, prop_input, text_embeds, text_input.attention_mask[:, 1:])
            prediction.append(output)
            output = model.property_embed(output.unsqueeze(2))
            prop_input = torch.cat([prop_input, output], dim=1)

        prediction = torch.stack(prediction, dim=-1)
        for i in range(len(data_loader)):
            gather.append(prediction[i].cpu()*std + mean)
        return gather

    reference, candidate = [], []
    for (prop, text) in data_loader:
        text_input = tokenizer(text, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(device)
        text_embeds = model.text_encoder.bert(text_input.input_ids[:, 1:], attention_mask=text_input.attention_mask[:, 1:],
                                              return_dict=True, mode='text').last_hidden_state
        prop_input = model.property_cls.expand(len(text), -1, -1)
        prediction = []
        for _ in range(53):
            output = generate(model, prop_input, text_embeds, text_input.attention_mask[:, 1:])
            prediction.append(output)
            output = model.property_embed(output.unsqueeze(2))
            prop_input = torch.cat([prop_input, output], dim=1)

        prediction = torch.stack(prediction, dim=-1)
        for i in range(prop.size(0)):
            reference.append(prop[i].cpu())
            candidate.append(prediction[i].cpu())
    print('SMILES-to-PV generation done')
    return reference, candidate


@torch.no_grad()
def metric_eval(ref, cand):
    with open('./normalize.pkl', 'rb') as w:
        norm = pickle.load(w)
    mean, std = norm
    mse = []
    n_mse = []
    rs, cs = [], []
    for i in range(len(ref)):
        r = (ref[i] * std) + mean
        c = (cand[i] * std) + mean
        rs.append(r)
        cs.append(c)
        mse.append((r - c) ** 2)
        n_mse.append((ref[i] - cand[i]) ** 2)
    mse = torch.stack(mse, dim=0)
    rmse = torch.sqrt(torch.mean(mse, dim=0)).squeeze()
    n_mse = torch.stack(n_mse, dim=0)
    n_rmse = torch.sqrt(torch.mean(n_mse, dim=0))
    print('mean of 53 properties\' normalized RMSE:', n_rmse.mean().item())

    rs = torch.stack(rs)
    cs = torch.stack(cs).squeeze()
    r2 = []
    for i in range(rs.size(1)):
        r2.append(r2_score(rs[:, i], cs[:, i]))
    r2 = np.array(r2)
    print('mean r^2 coefficient of determination:', r2.mean().item())

    plt.figure(figsize=(27, 27))
    with open('./property_name.txt', 'r') as f:
        names = [l.strip() for l in f.readlines()[:53]]
    for i, j in enumerate(range(53)):
        x = rs[:, j]
        y = cs[:, j]
        plt.subplot(9, 6, i + 1)
        plt.title(f'r2: {r2[j]:.3f}, RMSE: {rmse[j]:.3f}', fontdict={'fontsize': 9})
        plt.scatter(x, y, c='r', s=0.5, alpha=0.5, label=names[j])
        if i == 44:
            plt.xlim([-2, 2])
            plt.ylim([-2, 2])
        plt.rc('xtick', labelsize=7)
        plt.rc('ytick', labelsize=7)
        plt.legend(loc='upper left')
        plt.axline((0, 0), slope=1, color="#bbb", linestyle=(0, (5, 5)), zorder=-10)
    plt.show()


def main(args, config):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = random.randint(0, 1000)
    print('seed:', seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # === Dataset === #
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
            if 'queue' in key:
                del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)
    model = model.to(device)

    print("=" * 50)
    r_test, c_test = pv_generate(model, test_loader)
    metric_eval(r_test, c_test)
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./Pretrain/checkpoint_SPMM_20m.ckpt')
    parser.add_argument('--vocab_filename', default='./vocab_bpe_300.txt')
    parser.add_argument('--input_file', default='./data/3_SMILES2PV/zinc15_1k_unseen.txt')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    config = {
        'embed_dim': 256,
        'batch_size_test': 64,
        'bert_config_text': './config_bert.json',
        'bert_config_property': './config_bert_property.json',
    }
    main(args, config)
