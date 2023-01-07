import argparse
from pathlib import Path
import utils
import torch
from torch.distributions.categorical import Categorical
from bialbef import biALBEF
import time
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, WordpieceTokenizer
import datetime
from calc_property import calculate_property
from rdkit import Chem
import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')


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
def generate_with_property(model, property, tokenizer, device,n_sample,prop_mask):
    # test
    model.eval()
    print('Generating samples for evaluation...')
    start_time = time.time()

    with open('./normalize.pkl', 'rb') as w:
        norm = pickle.load(w)
    property_mean, property_std = norm
    property=(property-property_mean)/property_std
    prop=property.unsqueeze(0).repeat(n_sample,1)
    #prop+=torch.randn_like(prop)*0.1

    results=[]
    prop = prop.to(device, non_blocking=True)

    property1 = model.property_embed(prop.unsqueeze(2)) #batch*12*feature

    property_unk = model.property_mask.expand(property1.size(0), property1.size(1), -1)
    mpm_mask_expand = prop_mask.unsqueeze(0).unsqueeze(2).repeat(property_unk.size(0), 1, property_unk.size(2)).to(device)
    property_masked = property1 * (1 - mpm_mask_expand) + property_unk * mpm_mask_expand

    #property_masked=property1
    #mpm_mask_expand = prop_mask.unsqueeze(0).repeat(property1.size(0), 1)
    #mpm_mask_expand = torch.cat([torch.ones((property1.size(0),1)),mpm_mask_expand],dim=1).to(device)


    property = torch.cat([model.property_cls.expand(property_masked.size(0),-1,-1),property_masked],dim=1)
    prop_embeds = model.property_encoder(inputs_embeds=property,return_dict=True).last_hidden_state   #batch*len(=patch**2+1)*feature

    #text_input = torch.tensor([2,4]).expand(prop.size(0),2).to(device) #batch*2
    text_input = torch.tensor([2]).expand(prop.size(0), 1).to(device)  # batch*1

    for _ in range(100):
        output = generate(model,prop_embeds, text_input,stochastic=True)
        #output = generate(model,prop_embeds, text_input, stochastic=True, prop_att_mask=mpm_mask_expand)
        #text_input=torch.cat([text_input[:,:-1],output,4*torch.ones((output.size(0),1),dtype=torch.long).to(device)],dim=-1)
        text_input = torch.cat([text_input, output], dim=-1)
    for i in range(text_input.size(0)):
        sentence=text_input[i,1:]
        if 3 in sentence: sentence=sentence[:(sentence==3).nonzero(as_tuple=True)[0][0].item()]
        cdd=tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(sentence))
        results.append(cdd)

    print('time:',time.time()-start_time)
    return results


@torch.no_grad()
def metric_eval(prop_input,cand,mask):
    with open('./property_name.txt', 'r') as f: tmp=f.readlines()[1:54]
    names=[l.strip() for l in tmp]

    with open('./normalize.pkl', 'rb') as w:    norm = pickle.load(w)
    general_mean = norm[0]

    validity=0
    random.shuffle(cand)
    mse=[]
    valids=[]
    prop_cdds=[]
    for i in range(len(cand)):
        try:
            prop_cdd=calculate_property(cand[i][1:])
            mse.append((prop_input-prop_cdd)**2)
            prop_cdds.append(prop_cdd)
            validity+=1
            valids.append(cand[i])
            if validity==1: print('prop_cdd:', prop_cdd[[42, 41, 14, 50, 40, 45, 51, 20, 30, 31, 34, 52]], cand[i]) #print one sample
        except:
            continue
    mse=torch.stack(mse,dim=0)
    rmse=torch.sqrt(torch.mean(mse,dim=0))
    print('VALIDITY:',validity/(i+1))
    valids=[Chem.CanonSmiles(l[1:]) for l in valids]
    print('UNIQUENESS:',len(set(valids))/(i+1))
    #print('RMSE:',rmse)
    print("==============================================")
    #drawing graph
    prop_cdds=torch.stack(prop_cdds,dim=0)  #batch*12
    plt.figure(figsize=(12,12))
    kwargs = dict(hist_kws={'alpha': .4,'edgecolor':'#fff'}, kde_kws={'linewidth': 1})
    #for i in range(prop_cdds.size(-1)):
    for i,j in enumerate([42,41,14,50,40,45,51,20,30,31,34,52]):
        x=prop_cdds[:,j]
        plt.subplot(4, 3, i+1)
        color = "r" if mask[j]==0 else "royalblue"
        sns.distplot(x,color=color,bins=15,label=names[j],kde=True,**kwargs)
        plt.tick_params(left=False, labelleft=False)
        plt.xlim(norm[0][j]-3*norm[1][j],norm[0][j]+3*norm[1][j])
        plt.legend(loc='upper left')
        plt.margins(0.05, 0.2)
        if mask[j]==0:  plt.axvline(x=prop_input[j],linestyle="--",c='r',linewidth=1.5,zorder=-10)
        if mask[j]==0 or (1-mask).sum().item()==0:  plt.axvline(x=general_mean[j],linestyle="-", c='#666',linewidth=2)
    plt.suptitle("(d) No property control", fontsize=20,y=0.925)
    #plt.savefig('result2.png', dpi=150)    #save figure
    plt.show()

    # write generated molecules to txt file
    with open('generated_molecules.txt','w') as w:
        for v in valids:    w.write(v+'\n')
    return validity/(i+1)


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

    tokenizer = BertTokenizer(vocab_file='./vocab_bpe_300.txt', do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)

    #### Model ####
    print("Creating model")
    model = biALBEF(config=config, tokenizer=tokenizer)

    if args.checkpoint:
        print('LOADING PRETRAINED MODEL..')
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model = model.to(device)
    model_without_ddp = model

    start_time = time.time()

    print('GENERATE WITH SINGLE PROPERTY')

    ### Your have to set your prop_input and prop_mask from here ###
    prop_input=calculate_property('COc1cccc(NC(=O)CN(C)C(=O)COC(=O)c2cc(c3cccs3)nc3ccccc23)c1') # tensor with a length of 53
    prop_input[14] = 150        # 14th property: molecular weight
    
    prop_mask=torch.ones(53)    # In prop_mask, 1: masked, 0:not masked
    prop_mask[14]=0
    ### Your have to set your prop_input and prop_mask until here ###

    samples = generate_with_property(model_without_ddp, prop_input, tokenizer, device, args.n_generate, prop_mask)
    generate_stats = metric_eval(prop_input,samples,prop_mask)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./output/MG')
    #parser.add_argument('--checkpoint', default='./output/MG/checkpoint_best_bpe_nwp_pad0.5.pth')
    #parser.add_argument('--checkpoint', default='./output/MG/checkpoint_best_bpe_nwp_mask0.5.pth')
    parser.add_argument('--checkpoint', default='./Pretrain/checkpoint_08.pth')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', default=False)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--n_generate', default=100, type=int)
    args = parser.parse_args()

    cls_config = {
        'image_res': 224,
        'batch_size_train': 64,
        'batch_size_test': 16,
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
    }

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, cls_config)
