import argparse
from pathlib import Path
import utils
import torch
import numpy as np
from torch.distributions.categorical import Categorical
from bialbef import biALBEF
import time
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, WordpieceTokenizer
import datetime
from dataset import SMILESDataset_pretrain
from torch.utils.data import DataLoader
from calc_property import calculate_property
import random
import pickle


def generate(model, image_embeds, text, stochastic=True, prop_att_mask=None):
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
    model.eval()
    print('Generating samples for evaluation...')
    start_time = time.time()
    reference, candidate=[],[]
    for (prop,text) in data_loader:
        prop = prop.to(device, non_blocking=True)

        property1 = model.property_embed(prop.unsqueeze(2)) #batch*12*feature
        property = torch.cat([model.property_cls.expand(property1.size(0),-1,-1),property1],dim=1)
        prop_embeds = model.property_encoder(inputs_embeds=property,return_dict=True).last_hidden_state   #batch*len(=patch**2+1)*feature

        #text_input = torch.tensor([2,4]).expand(prop.size(0),2).to(device) #batch*2
        text_input = torch.tensor([2]).expand(prop.size(0), 1).to(device)  # batch*1

        for _ in range(100):
            output = generate(model,prop_embeds, text_input,stochastic=False)
            #text_input=torch.cat([text_input[:,:-1],output,4*torch.ones((output.size(0),1),dtype=torch.long).to(device)],dim=-1)
            text_input = torch.cat([text_input, output], dim=-1)
        for i in range(text_input.size(0)):
            reference.append(text[i])
            sentence=text_input[i,1:]
            if 3 in sentence: sentence=sentence[:(sentence==3).nonzero(as_tuple=True)[0][0].item()]
            cdd=tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(sentence))
            candidate.append(cdd)
    print('time:',time.time()-start_time)
    return reference,candidate


@torch.no_grad()
def metric_eval(ref,cand):
    with open('./normalize.pkl', 'rb') as w:    norm = pickle.load(w)
    property_mean,property_std = norm
    #property_mean = torch.tensor([1.4267, 4.2140, 363.0794, 2.7840, 1.9534, 5.6722, 71.2552,
    #                                   25.0858, 26.8583, 2.7226, 96.8194, 0.6098])
    #property_std = torch.tensor([1.7206, 2.7012, 164.6371, 1.6557, 1.4255, 5.3647, 54.3515,
    #                                  11.7913, 12.8683, 2.7610, 44.8578, 0.2197])
    validity=0
    valids=[]
    mse=[]
    n_mse=[]
    for i in range(len(ref)):
        try:
            prop_ref=calculate_property(ref[i][1:])
            prop_cdd=calculate_property(cand[i][1:])
            n_ref=(prop_ref-property_mean)/property_std
            n_cdd=(prop_cdd - property_mean) / property_std
            mse.append((prop_ref-prop_cdd)**2)
            n_mse.append((n_ref - n_cdd) ** 2)
            validity+=1
            valids.append(cand[i][1:])
            if validity==72:
                print('prop_ref:', prop_ref, ref[i])
                print('prop_cdd:', prop_cdd, cand[i])
        except:
            continue
    if len(mse)!=0:
        mse=torch.stack(mse,dim=0)
        rmse=torch.sqrt(torch.mean(mse,dim=0))
        n_mse=torch.stack(n_mse,dim=0)
        n_rmse=torch.sqrt(torch.mean(n_mse,dim=0))
        n_rmse=n_rmse[property_std!=0]
    else:   rmse,n_rmse=0,0
    print('VALIDITY:',validity/len(ref))
    #print('RMSE:',rmse)
    print('N_RMSE:', n_rmse.mean(),n_rmse.max())
    print("==============================================")

    with open('generated_molecules.txt','w') as w:
        for v in valids:    w.write(v+'\n')

    return validity/len(ref)


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
    parser.add_argument('--output_dir', default='./output/MG')
    #parser.add_argument('--checkpoint', default='./Pretrain/checkpoint_05_noaug.pth')
    #parser.add_argument('--checkpoint', default='')
    parser.add_argument('--checkpoint', default='./Pretrain/checkpoint_08.pth')
    parser.add_argument('--vocab_filename', default='./vocab_bpe_300.txt')
    parser.add_argument('--evaluate', default=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=43, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    cls_config = {
        'image_res': 224,
        'batch_size_train': 32,
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