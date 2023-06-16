import copy
from xbert import BertConfig, BertForMaskedLM
import torch
from torch import nn
from torch.distributions.categorical import Categorical


class SPMM_rxn(nn.Module):
    def __init__(self, cp=None, config=None,):
        super().__init__()

        self.text_encoder = BertForMaskedLM(config=BertConfig.from_json_file(config['bert_config_text']))
        self.text_encoder2 = BertForMaskedLM(config=BertConfig.from_json_file(config['bert_config_smiles']))

        # copy weights of checkpoint's SMILES encoder to text_encoder2
        if cp:
            checkpoint = torch.load(cp, map_location='cpu')
            try:
                state_dict = copy.deepcopy(checkpoint['model'])
            except:
                state_dict = copy.deepcopy(checkpoint['state_dict'])
            for key in list(state_dict.keys()):
                if 'text_encoder.' in key:
                    new_key = key.replace('text_encoder.', '')
                    state_dict[new_key] = state_dict[key]
                del state_dict[key]
            msg = self.text_encoder2.load_state_dict(state_dict, strict=False)
            # print(msg)
            del state_dict

    def forward(self, text_input_ids, text_attention_mask, product_input_ids, product_attention_mask):
        input_ids = product_input_ids.clone()
        labels = input_ids.clone()[:, 1:]
        text_embeds = self.text_encoder2.bert(text_input_ids, attention_mask=text_attention_mask, return_dict=True, mode='text').last_hidden_state
        mlm_output = self.text_encoder(input_ids,
                                       attention_mask=product_attention_mask,
                                       encoder_hidden_states=text_embeds,
                                       encoder_attention_mask=text_attention_mask,
                                       return_dict=True,
                                       is_decoder=True,
                                       return_logits=True,
                                       )[:, :-1, :]

        loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        loss_mlm = loss_fct(mlm_output.permute((0, 2, 1)), labels)
        return loss_mlm

    def generate(self, text_embeds, text_mask, product_input, stochastic=False, k=None):
        product_atts = torch.where(product_input == 0, 0, 1)

        token_output = self.text_encoder(product_input,
                                         attention_mask=product_atts,
                                         encoder_hidden_states=text_embeds,
                                         encoder_attention_mask=text_mask,
                                         return_dict=True,
                                         is_decoder=True,
                                         return_logits=True,
                                         )[:, -1, :]  # batch*300
        if k:
            p = torch.softmax(token_output, dim=-1)
            output = torch.topk(p, k=k, dim=-1)  # batch*k
            return torch.log(output.values), output.indices
        if stochastic:
            p = torch.softmax(token_output, dim=-1)
            m = Categorical(p)
            token_output = m.sample()
        else:
            token_output = torch.argmax(token_output, dim=-1)
        return token_output.unsqueeze(1)  # batch*1
