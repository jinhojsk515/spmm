import argparse
from tqdm import tqdm
import torch
import numpy as np
import time
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, WordpieceTokenizer
import datetime
from dataset import SMILESDataset_LIPO, SMILESDataset_BACER, SMILESDataset_Clearance, SMILESDataset_ESOL, SMILESDataset_Freesolv
from torch.utils.data import DataLoader
import torch.optim as optim
from scheduler import create_scheduler
import random
import torch.nn as nn
from xbert import BertConfig, BertForMaskedLM


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class SPMM_regressor(nn.Module):
    def __init__(self, tokenizer=None, config=None):
        super().__init__()
        self.tokenizer = tokenizer

        bert_config = BertConfig.from_json_file(config['bert_config_text'])
        self.text_encoder = BertForMaskedLM(config=bert_config)
        for i in range(bert_config.fusion_layer, bert_config.num_hidden_layers):  self.text_encoder.bert.encoder.layer[i] = nn.Identity()
        self.text_encoder.cls = nn.Identity()
        text_width = self.text_encoder.config.hidden_size

        self.reg_head = nn.Sequential(
            nn.Linear(text_width * 1, text_width * 2),
            nn.GELU(),
            nn.Linear(text_width * 2, 1)
        )

    def forward(self, text_input_ids, text_attention_mask, value, eval=False):
        vl_embeddings = self.text_encoder.bert(text_input_ids, attention_mask=text_attention_mask, return_dict=True, mode='text').last_hidden_state
        vl_embeddings = vl_embeddings[:, 0, :]
        pred = self.reg_head(vl_embeddings).squeeze(-1)

        if eval:    return pred
        lossfn = nn.MSELoss()
        loss = lossfn(pred, value)
        return loss


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler):
    # train
    model.train()

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 20
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    tqdm_data_loader = tqdm(data_loader, miniters=print_freq, desc=header)
    for i, (text, value) in enumerate(tqdm_data_loader):
        optimizer.zero_grad()
        value = value.to(device, non_blocking=True)

        text_input = tokenizer(text, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(device)

        loss = model(text_input.input_ids[:, 1:], text_input.attention_mask[:, 1:], value)
        loss.backward()
        optimizer.step()

        tqdm_data_loader.set_description(f'loss={loss.item():.4f}, lr={optimizer.param_groups[0]["lr"]:.6f}')

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, denormalize=None):
    # test
    model.eval()
    preds = []
    answers = []
    for text, value in data_loader:
        value = value.to(device, non_blocking=True)
        text_input = tokenizer(text, padding='longest', return_tensors="pt").to(device)

        prediction = model(text_input.input_ids[:, 1:], text_input.attention_mask[:, 1:], value, eval=True)

        preds.append(prediction.cpu())
        answers.append(value.cpu())

    preds = torch.cat(preds, dim=0)
    answers = torch.cat(answers, dim=0)

    value_mean = denormalize[0]
    value_std = denormalize[1]

    preds = preds * value_std + value_mean
    answers = answers * value_std + value_mean
    lossfn = nn.MSELoss()
    return torch.sqrt(lossfn(preds, answers)).item()


def main(args, config):
    device = torch.device(args.device)
    print('DATASET:', args.name)

    # === Dataset === #
    print("Creating dataset")
    name = args.name
    if name == 'bace':
        dataset_train = SMILESDataset_BACER('data/4_MoleculeNet/BACER_train.csv')
        dataset_val = SMILESDataset_BACER('data/4_MoleculeNet/BACER_valid.csv')
        dataset_test = SMILESDataset_BACER('data/4_MoleculeNet/BACER_test.csv')
    elif name == 'lipo':
        dataset_train = SMILESDataset_LIPO('data/4_MoleculeNet/LIPO_train.csv')
        dataset_val = SMILESDataset_LIPO('data/4_MoleculeNet/LIPO_valid.csv')
        dataset_test = SMILESDataset_LIPO('data/4_MoleculeNet/LIPO_test.csv')
    elif name == 'esol':
        dataset_train = SMILESDataset_ESOL('data/4_MoleculeNet/ESOL_train.csv')
        dataset_val = SMILESDataset_ESOL('data/4_MoleculeNet/ESOL_valid.csv')
        dataset_test = SMILESDataset_ESOL('data/4_MoleculeNet/ESOL_test.csv')
    elif name == 'freesolv':
        dataset_train = SMILESDataset_Freesolv('data/4_MoleculeNet/freesolv_train.csv')
        dataset_val = SMILESDataset_Freesolv('data/4_MoleculeNet/freesolv_valid.csv')
        dataset_test = SMILESDataset_Freesolv('data/4_MoleculeNet/freesolv_test.csv')
    elif name == 'clearance':
        dataset_train = SMILESDataset_Clearance('data/4_MoleculeNet/Clearance_train.csv')
        dataset_val = SMILESDataset_Clearance('data/4_MoleculeNet/Clearance_valid.csv')
        dataset_test = SMILESDataset_Clearance('data/4_MoleculeNet/Clearance_test.csv')

    print(len(dataset_train), len(dataset_val), len(dataset_test))
    train_loader = DataLoader(dataset_train, batch_size=config['batch_size_train'], num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset_val, batch_size=config['batch_size_test'], num_workers=8, pin_memory=True, drop_last=False)
    test_loader = DataLoader(dataset_test, batch_size=config['batch_size_test'], num_workers=8, pin_memory=True, drop_last=False)

    tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)

    # fix the seed for reproducibility
    seed = args.seed if args.seed else random.randint(0, 100)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # === Model === #
    print("Creating model")
    model = SPMM_regressor(config=config, tokenizer=tokenizer)
    print('#parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.checkpoint:
        print('LOADING PRETRAINED MODEL..')
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['state_dict']
        for key in list(state_dict.keys()):
            if '_unk' in key:
                new_key = key.replace('_unk', '_mask')
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        # print(msg)

    model = model.to(device)
    arg_opt = config['optimizer']
    optimizer = optim.AdamW(model.parameters(), lr=arg_opt['lr'], weight_decay=arg_opt['weight_decay'])

    arg_sche = AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best_valid = 10000.
    best_test = 0

    start_time = time.time()

    for epoch in range(0, max_epoch):
        print('TRAIN', epoch)
        train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler)
        denormalize = (dataset_train.value_mean, dataset_train.value_std)
        val_stats = evaluate(model, val_loader, tokenizer, device, denormalize)
        print('VALID MSE: %.4f' % val_stats)
        test_stats = evaluate(model, test_loader, tokenizer, device, denormalize)
        print('TEST MSE: %.4f' % test_stats)

        if val_stats < best_valid:
            best_valid = val_stats
            best_test = test_stats
        lr_scheduler.step(epoch + warmup_steps + 1)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('DATASET:', args.name, '\tTest set MSE of the checkpoint with best validation MSE:', best_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./Pretrain/checkpoint_SPMM.ckpt')
    parser.add_argument('--vocab_filename', default='./vocab_bpe_300.txt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=40, type=int)
    parser.add_argument('--name', default='bace', type=str)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--min_lr', default=3e-6, type=float)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    args = parser.parse_args()

    cls_config = {
        'batch_size_train': args.batch_size,
        'batch_size_test': 16,
        'embed_dim': 256,
        'bert_config_text': './config_bert.json',
        'bert_config_property': './config_bert_property.json',
        'schedular': {'sched': 'cosine', 'lr': args.lr, 'epochs': args.epoch, 'min_lr': args.min_lr,
                      'decay_rate': 1, 'warmup_lr': 0.5e-5, 'warmup_epochs': 1, 'cooldown_epochs': 0},
        'optimizer': {'opt': 'adamW', 'lr': args.lr, 'weight_decay': 0.02}
    }
    main(args, cls_config)
