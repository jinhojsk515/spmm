import argparse
from tqdm import tqdm
from pathlib import Path
import utils
import torch
import numpy as np
from bialbef import biALBEF
import time
import os
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, WordpieceTokenizer
import datetime
from dataset import SMILESDataset_pretrain
from torch.utils.data import DataLoader
import torch.optim as optim
from scheduler import create_scheduler
import random
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import r2_score


def generate(model, prop_input, text_embeds, text_atts, stochastic=True):
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
    pred = model.property_mtr_head(token_output).squeeze(-1)[:, -1]  # batch
    return pred.unsqueeze(1)  # batch*1


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device):
    # test
    model.eval()
    print('Generating samples for evaluation...')
    start_time = time.time()
    reference, candidate=[],[]
    for (prop,text) in data_loader:
        text_input = tokenizer(text, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(device)
        text_embeds = model.text_encoder.bert(text_input.input_ids, attention_mask=text_input.attention_mask, return_dict=True, mode='text').last_hidden_state
        prop_input = model.property_cls.expand(len(text),-1,-1)  #batch*1*feature
        prediction=[]
        for _ in range(53):
            output = generate(model,prop_input, text_embeds, text_input.attention_mask, stochastic=False)
            #text_input=torch.cat([text_input[:,:-1],output,4*torch.ones((output.size(0),1),dtype=torch.long).to(device)],dim=-1)
            prediction.append(output)
            output = model.property_embed(output.unsqueeze(2))  # batch*1*feature
            prop_input = torch.cat([prop_input, output], dim=1)

        prediction=torch.stack(prediction,dim=-1)   #batch*12
        for i in range(prop.size(0)):
            reference.append(prop[i].cpu())
            candidate.append(prediction[i].cpu())
    print('time:',time.time()-start_time)
    return reference,candidate


@torch.no_grad()
def metric_eval(ref,cand):
    #mean = torch.tensor([1.4267, 4.2140, 363.0794, 2.7840, 1.9534, 5.6722, 71.2552,
    #                                   25.0858, 26.8583, 2.7226, 96.8194, 0.6098])
    #std = torch.tensor([1.7206, 2.7012, 164.6371, 1.6557, 1.4255, 5.3647, 54.3515,
    #                                  11.7913, 12.8683, 2.7610, 44.8578, 0.2197])
    with open('./property_name.txt', 'r') as f: tmp=f.readlines()[1:54]
    names=[l.strip() for l in tmp]
    with open('./normalize.pkl', 'rb') as w:    norm = pickle.load(w)
    mean,std = norm
    mse=[]
    n_mse=[]
    rs,cs=[],[]
    for i in range(len(ref)):
        r = (ref[i] * std) + mean
        c = (cand[i] * std) + mean
        rs.append(r)
        cs.append(c)
        mse.append((r-c)**2)
        n_mse.append((ref[i]-cand[i])**2)
    mse=torch.stack(mse,dim=0)
    rmse=torch.sqrt(torch.mean(mse,dim=0)).squeeze()
    #print('RMSE:',rmse)
    nmse = torch.stack(n_mse, dim=0)
    rnmse = torch.sqrt(torch.mean(nmse, dim=0))
    print('N_RMSE:',rnmse.mean(),rnmse.max())

    rs = torch.stack(rs)    #batch*12
    cs = torch.stack(cs).squeeze()
    r2=[]
    for i in range(rs.size(1)):
        r2.append(r2_score(rs[:,i],cs[:,i]))
    r2=np.array(r2)
    count=0
    for r in r2:
        if r>0.98:  count+=1
    print('R2:',r2.mean(),r2[52],count)

    plt.figure(figsize=(13,10))
    for i, j in enumerate([42, 41, 14, 50, 40, 45, 51, 20, 30, 31, 34, 52]):
    #for i in range(rs.size(-1)):
        x=rs[:,j]
        y=cs[:,j]
        plt.subplot(4, 3, i+1)
        plt.title(f'r2: {r2[j]:.3f}, RMSE: {rmse[j]:.3f}', fontdict={'fontsize': 7},y=0.8)
        plt.scatter(x,y,c='r',s=1,label=names[j])
        plt.rc('xtick', labelsize=7)
        plt.rc('ytick', labelsize=7)
        plt.legend()
        plt.axline((0, 0), slope=1, color="#bbb", linestyle=(0, (5, 5)),zorder=-10)
    #plt.savefig('result1.png', dpi=300)    #save figure
    plt.show()
    ''' #supplement figure
    plt.figure(figsize=(27,27))
    for i, j in enumerate(range(53)):
    #for i in range(rs.size(-1)):
        x=rs[:,j]
        y=cs[:,j]
        plt.subplot(9, 6, i+1)
        plt.title(f'r2: {r2[j]:.3f}, RMSE: {rmse[j]:.3f}',fontdict = {'fontsize' : 9})
        plt.scatter(x,y,c='r',s=0.5,alpha=0.5,label=names[j])
        plt.rc('xtick', labelsize=7)
        plt.rc('ytick', labelsize=7)
        plt.legend(loc='upper left')
        plt.axline((0, 0), slope=1, color="#bbb", linestyle=(0, (5, 5)),zorder=-10)
    plt.savefig('supp1.png', dpi=300)
    plt.show()
    '''
    print("==============================================")

    return rmse.mean()


def main(args, config):
    print('Evaluation mode:', args.evaluate)
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    #seed = args.seed + utils.get_rank()
    seed=random.randint(0,1000)
    print('seed:',seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating dataset")
    dataset_test=SMILESDataset_pretrain('./data/pubchem-10m-simple.txt',data_length=[2000000,2000100])
    test_loader = DataLoader(dataset_test, batch_size=config['batch_size_test'], pin_memory=True, drop_last=False)

    tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)

    #### Model ####
    print("Creating model")
    model = biALBEF(config=config, tokenizer=tokenizer)

    if args.checkpoint:
        print('LOADING PRETRAINED MODEL..')
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        for key in list(state_dict.keys()):
            if '_unk' in key:
                new_key = key.replace('_unk', '_mask')
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model = model.to(device)
    model_without_ddp = model

    start_time = time.time()

    r_test,c_test = evaluate(model_without_ddp, test_loader, tokenizer, device)
    print('TEST')
    test_stats = metric_eval(r_test,c_test)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./output/PG')
    #parser.add_argument('--checkpoint', default='./output/PG/checkpoint_best.pth')
    #parser.add_argument('--checkpoint', default='')
    parser.add_argument('--checkpoint', default='./Pretrain/checkpoint_08.pth')
    parser.add_argument('--vocab_filename', default='./vocab_bpe_300.txt')
    parser.add_argument('--evaluate', default=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    cls_config = {
        'image_res': 224,
        'batch_size_train': 64,
        'batch_size_test': 4,
        'alpha': 0.4,
        'queue_size': 4096,    #65536
        'momentum': 0.995,
        'property_width': 384,
        'embed_dim': 256,
        'temp': 0.07,
        'mlm_probability': 0.15,
        'distill':True,
        'warm_up':True,
        'bert_config_text': './config_bert.json',
        'bert_config_property': './config_bert_property.json',
        'schedular': {'sched': 'cosine', 'lr': 3e-5, 'epochs': 10, 'min_lr': 1e-6,
                      'decay_rate': 1, 'warmup_lr': 1e-5, 'warmup_epochs': 1, 'cooldown_epochs': 0},
        'optimizer': {'opt': 'adamW', 'lr': 3e-5, 'weight_decay': 0.02}
    }

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args, cls_config)
#vocab file,config_bert.vocab_size, savefile_name
#evaluate, import location