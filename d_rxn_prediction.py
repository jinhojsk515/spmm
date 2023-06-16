import argparse
from tqdm import tqdm
from pathlib import Path
import torch
import numpy as np
import random
from SPMM_models_rxn import SPMM_rxn
import time
import os
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, WordpieceTokenizer
import datetime
from dataset import SMILESDataset_USPTO, SMILESDataset_USPTO_reverse
from torch.utils.data import DataLoader
import torch.optim as optim
from scheduler import create_scheduler
from rdkit import Chem
from rdkit import RDLogger


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler):
    # train
    model.train()

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    tqdm_data_loader = tqdm(data_loader, miniters=print_freq, desc=header)
    losses = np.array([0.])
    for i, (text, product) in enumerate(tqdm_data_loader):
        text_input = tokenizer(text, padding='longest', max_length=150, return_tensors="pt").to(device)
        product_input = tokenizer(product, padding='longest', max_length=100, return_tensors="pt").to(device)

        loss = model(text_input.input_ids[:, 1:], text_input.attention_mask[:, 1:], product_input.input_ids[:, 1:], product_input.attention_mask[:, 1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses = losses * 0.99 + 0.01 * np.array([loss.item()])
        tqdm_data_loader.set_description(f'loss={loss.item():.4f}, lr={optimizer.param_groups[0]["lr"]:.6f}')

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)
    print('mean loss:', losses)


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device):
    # test
    model.eval()

    reference, candidate = [], []
    for (text, product) in data_loader:
        product_input = torch.tensor([tokenizer.cls_token_id]).expand(len(text), 1).to(device)  # batch*1
        text_input = tokenizer(text, padding='longest', max_length=150, return_tensors="pt").to(device)
        text_embeds = model.text_encoder2.bert(text_input.input_ids[:, 1:], attention_mask=text_input.attention_mask[:, 1:], return_dict=True,
                                               mode='text').last_hidden_state
        end_count = torch.zeros_like(product_input).to(bool)
        for _ in range(100):
            output = model.generate(text_embeds, text_input.attention_mask[:, 1:], product_input, stochastic=False)
            end_count = torch.logical_or(end_count, (output == tokenizer.sep_token_id))
            if end_count.all():
                break
            product_input = torch.cat([product_input, output], dim=-1)
        for i in range(product_input.size(0)):
            reference.append(product[i].replace('[CLS]', ''))

            sentence = product_input[i]
            if tokenizer.sep_token_id in sentence: sentence = sentence[:(sentence == tokenizer.sep_token_id).nonzero(as_tuple=True)[0][0].item()]
            cdd = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(sentence)).replace('[CLS]', '')

            candidate.append(cdd)
    return reference, candidate


# beam search
@torch.no_grad()
def evaluate_beam(model, data_loader, tokenizer, device, k=3):
    # test
    model.eval()
    reference, candidate = [], []
    for (text, product) in tqdm(data_loader):

        text_input = tokenizer(text, padding='longest', max_length=150, return_tensors="pt").to(device)
        text_embeds = model.text_encoder2.bert(text_input.input_ids[:, 1:], attention_mask=text_input.attention_mask[:, 1:], return_dict=True,
                                               mode='text').last_hidden_state
        product_input = torch.tensor([tokenizer.cls_token_id]).expand(1, 1).to(device)
        values, indices = model.generate(text_embeds, text_input.attention_mask[:, 1:], product_input, stochastic=False, k=k)
        product_input = torch.cat([torch.tensor([tokenizer.cls_token_id]).expand(k, 1).to(device), indices.squeeze(0).unsqueeze(-1)], dim=-1)
        current_p = values.squeeze(0)
        final_output = []
        for _ in range(100):
            values, indices = model.generate(text_embeds, text_input.attention_mask[:, 1:], product_input, stochastic=False, k=k)
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

        reference.append(product[0].replace('[CLS]', ''))
        candidate_k = []
        final_output = sorted(final_output, key=lambda x: x[0], reverse=True)[:k]
        for p, sentence in final_output:
            cdd = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(sentence[:-1])).replace('[CLS]', '')
            candidate_k.append(cdd)
        candidate.append(candidate_k)
    return reference, candidate


@torch.no_grad()
def metric_eval(ref, cand):
    correct = 0
    RDLogger.DisableLog('rdApp.*')
    for i in range(len(ref)):
        try:
            r = Chem.MolToSmiles(Chem.MolFromSmiles(ref[i]), isomericSmiles=False, canonical=True)
            if type(cand[i]) is str:
                c = Chem.MolToSmiles(Chem.MolFromSmiles(cand[i]), isomericSmiles=False, canonical=True)
                if r == c:    correct += 1
            else:
                for c in cand[i]:
                    c = Chem.MolToSmiles(Chem.MolFromSmiles(c), isomericSmiles=False, canonical=True)
                    if r == c:
                        correct += 1
                        break
        except:
            continue
    print('Accuracy:', correct / len(ref))
    return correct / len(ref)


def main(args, config):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = random.randint(0, 1000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # === Dataset === #
    print("Creating dataset")
    if args.mode == 'forward':
        dataset_train = SMILESDataset_USPTO('./data/6_RXNprediction/USPTO-480k/train_parsed2.txt', data_length=None, aug=True)
        dataset_val = SMILESDataset_USPTO('./data/6_RXNprediction/USPTO-480k/valid_parsed2.txt', data_length=[0, 100])
        dataset_test = SMILESDataset_USPTO('./data/6_RXNprediction/USPTO-480k/test_parsed2.txt', data_length=None)
    elif args.mode == 'retro':
        dataset_train = SMILESDataset_USPTO_reverse(mode='train', data_length=None, aug=True)
        dataset_val = SMILESDataset_USPTO_reverse(mode='test', data_length=[0, 100])
        dataset_test = SMILESDataset_USPTO_reverse(mode='test', data_length=None)
    else:
        print("\'args.mode\' should be \'forward\' or \'retro\'")
        raise NotImplementedError

    print(len(dataset_train), len(dataset_val), len(dataset_test))
    train_loader = DataLoader(dataset_train, batch_size=config['batch_size_train'], num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset_val, batch_size=config['batch_size_test'], num_workers=8, pin_memory=True, drop_last=False)
    test_loader = DataLoader(dataset_test, batch_size=config['batch_size_test'], num_workers=8, pin_memory=True, drop_last=False)

    tokenizer = BertTokenizer(vocab_file='vocab_bpe_300.txt', do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)

    # === Model === #
    print("Creating model")
    model = SPMM_rxn(config=config, cp=args.checkpoint)
    print('#parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint['state_dict']

        for key in list(state_dict.keys()):
            if 'queue' in key or 'property' in key or '_m' in key:
                del state_dict[key]
            if '_unk' in key:
                new_key = key.replace('_unk', '_mask')
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)
    model = model.to(device)

    arg_opt = config['optimizer']
    optimizer = optim.AdamW(model.parameters(), lr=arg_opt['lr'], weight_decay=arg_opt['weight_decay'])

    arg_sche = AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best_valid = 0
    best_test = 0

    start_time = time.time()

    for epoch in range(0, max_epoch):
        if not args.evaluate:
            print('TRAIN', epoch)
            train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler)
        print('VALIDATION')
        if args.n_beam == 1:
            r_valid, c_valid = evaluate(model, val_loader, tokenizer, device)
        else:
            r_valid, c_valid = evaluate_beam(model, val_loader, tokenizer, device, k=args.n_beam)
        val_stats = metric_eval(r_valid, c_valid)
        print('TEST')
        if args.n_beam == 1:
            r_test, c_test = evaluate(model, test_loader, tokenizer, device)
        else:
            r_test, c_test = evaluate_beam(model, test_loader, tokenizer, device, k=args.n_beam)
        test_stats = metric_eval(r_test, c_test)

        if not args.evaluate:
            if val_stats >= best_valid:
                save_obj = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                print('SAVING...', test_stats)
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                best_valid = val_stats
                best_test = test_stats
        if args.evaluate:   break
        lr_scheduler.step(epoch + warmup_steps + 1)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('test ACC of checkpoint with best val ACC:', best_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./output/RXN')
    # parser.add_argument('--checkpoint', default='./Pretrain/checkpoint_SPMM_20m.pth')
    parser.add_argument('--checkpoint', default='./output/RXN/checkpoint_forward_best.pth')
    # parser.add_argument('--checkpoint', default='./output/RXN/checkpoint_best.pth')
    parser.add_argument('--mode', default='forward', type=str)          # 'forward' or 'retro'
    parser.add_argument('--evaluate', default=True, type=bool)          # if True, only evaluate the model on valid&test set (skip training)
    parser.add_argument('--n_beam', default=5, type=int)                # if >1, use beam search to generate output
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', default=5e-6, type=float)
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    args = parser.parse_args()

    cls_config = {
        'batch_size_train': args.batch_size,
        'batch_size_test': 1 if args.n_beam != 1 else 32,
        'bert_config_text': './config_bert.json',
        'bert_config_smiles': './config_bert_smiles.json',
        'schedular': {'sched': 'cosine', 'lr': args.lr, 'epochs': args.epoch, 'min_lr': args.min_lr,
                      'decay_rate': 1, 'warmup_lr': 1e-5, 'warmup_epochs': 1, 'cooldown_epochs': 0},
        'optimizer': {'opt': 'adamW', 'lr': args.lr, 'weight_decay': 0.02}
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, cls_config)
