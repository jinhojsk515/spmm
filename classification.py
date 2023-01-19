import argparse
from tqdm import tqdm
from pathlib import Path
import utils
import torch
import numpy as np
import time
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, WordpieceTokenizer
import datetime
from dataset import SMILESDataset_BBBP
from torch.utils.data import DataLoader
import torch.optim as optim
from scheduler import create_scheduler
import torch.nn.functional as F
import random
from sklearn.metrics import f1_score, roc_auc_score
import torch.nn as nn
from xbert import BertConfig, BertForMaskedLM


class Classifier(nn.Module):
    def __init__(self, config=None,):
        super().__init__()
        bert_config = BertConfig.from_json_file(config['bert_config_text'])
        self.text_encoder = BertForMaskedLM(config=bert_config)
        text_width = self.text_encoder.config.hidden_size

        # The fusion encoder and the token prediction module are not used, so we can remove them
        for i in [4, 5, 6, 7]:
            self.text_encoder.bert.encoder.layer[i] = nn.Identity()
        self.text_encoder.cls = nn.Identity()

        self.reg_head = nn.Sequential(
            nn.Linear(text_width, 2048),
            nn.GELU(),
            nn.Linear(2048, 2)
        )

    def forward(self, text_input_ids, text_attention_mask, value, evaluation=False):  # text=(batch*len,batch*len)
        vl_embeddings = self.text_encoder.bert(text_input_ids, attention_mask=text_attention_mask, return_dict=True, mode='text').last_hidden_state[:, 0, :]
        pred = self.reg_head(vl_embeddings)    # batch*feature => batch*2
        if evaluation:
            return pred

        loss = F.cross_entropy(pred, value)
        return loss


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler):
    # train
    model.train()

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 50
    warmup_iterations = warmup_steps * step_size

    tqdm_data_loader = tqdm(data_loader, miniters=print_freq, desc=header)
    losses = []
    for i, (text, label) in enumerate(tqdm_data_loader):
        optimizer.zero_grad()
        label = label.to(device, non_blocking=True)
        text_input = tokenizer(text, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(device)

        loss = model(text_input.input_ids, text_input.attention_mask, label)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        tqdm_data_loader.set_description(f'loss={loss.item():.4f}, lr={optimizer.param_groups[0]["lr"]:.6f}')

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)
    print('mean loss:%.5f' % np.array(losses).mean())


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device):
    # test
    model.eval()
    scores = []
    preds = []
    answers = []
    for text, label in data_loader:
        label = label.to(device, non_blocking=True)
        text_input = tokenizer(text, padding='longest', return_tensors="pt").to(device)
        prediction = model(text_input.input_ids, text_input.attention_mask, label, evaluation=True)

        score = torch.softmax(prediction, dim=-1)
        prediction = torch.argmax(score, dim=-1)
        scores.append(score[:, 1].cpu())
        preds.append(prediction.cpu())
        answers.append(label.cpu())

    scores = torch.cat(scores, dim=0)
    preds = torch.cat(preds, dim=0)
    answers = torch.cat(answers, dim=0)

    print('F1 score:', f1_score(answers, preds))
    print('AUROC:', roc_auc_score(answers, scores))
    print('Accuracy:', ((answers == preds).sum() / answers.numel()).item())

    score = roc_auc_score(answers, scores)
    return score


def main(args, config):
    print('Evaluation mode:', args.evaluate)
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # ### Dataset ### #
    print("Creating dataset")
    dataset_train = SMILESDataset_BBBP('./data/BBBP_train.csv')
    dataset_val = SMILESDataset_BBBP('./data/BBBP_valid.csv')
    dataset_test = SMILESDataset_BBBP('./data/BBBP_test.csv')
    print(len(dataset_train), len(dataset_val), len(dataset_test))
    train_loader = DataLoader(dataset_train, batch_size=config['batch_size_train'], num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset_val, batch_size=config['batch_size_test'], num_workers=8, pin_memory=True, drop_last=False)
    test_loader = DataLoader(dataset_test, batch_size=config['batch_size_test'], num_workers=8, pin_memory=True, drop_last=False)

    tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)

    # ### Model ### #
    print("Creating model")
    model = Classifier(config=config)
    print('#parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.checkpoint:
        print('LOADING PRETRAINED MODEL..')
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        # print(msg)
    model = model.to(device)

    arg_opt = config['optimizer']
    optimizer = optim.AdamW(model.parameters(), lr=arg_opt['lr'], weight_decay=arg_opt['weight_decay'])

    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0.
    best_test = 0.

    start_time = time.time()

    for epoch in range(0, max_epoch):
        if not args.evaluate:
            print('TRAIN', epoch)
            train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler)
        # print('VALIDATION')
        score_valid = evaluate(model, val_loader, tokenizer, device)
        print('VALID\t%.4f' % score_valid)
        # print('TEST')
        score_test = evaluate(model, test_loader, tokenizer, device)
        print('TEST\t%.4f' % score_test)

        if not args.evaluate:
            if score_valid >= best:
                best = score_valid
                best_test = score_test
                print('new test set result', score_test)
        if args.evaluate:
            break
        lr_scheduler.step(epoch + warmup_steps + 1)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('BEST PERFORMANCE:', best_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./output/CLS')
    parser.add_argument('--checkpoint', default='./Pretrain/checkpoint_08.pth')
    parser.add_argument('--vocab_filename', default='./vocab_bpe_300.txt')
    parser.add_argument('--evaluate', default=False)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=43, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    cls_config = {
        'batch_size_train': 16,  # 4 for bace, 16 for bbbp, (32 for hiv?)
        'batch_size_test': 64,
        'bert_config_text': './config_bert.json',
        'schedular': {'sched': 'cosine', 'lr': 2.5e-5, 'epochs': 20, 'min_lr': 5e-6,
                      'decay_rate': 1, 'warmup_lr': 1e-5, 'warmup_epochs': 1, 'cooldown_epochs': 0},
        'optimizer': {'opt': 'adamW', 'lr': 2.5e-5, 'weight_decay': 0.02}
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, cls_config)
# bbbp : 16,2.5e-5,20,5e-6
