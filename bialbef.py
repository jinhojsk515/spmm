from xbert import BertConfig, BertForMaskedLM
import torch
import torch.nn.functional as F
from torch import nn


class biALBEF(nn.Module):
    def __init__(self,
                 tokenizer=None,
                 config=None,
                 temp=0.07,
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.mlm_probability = config['mlm_probability']
        embed_dim = config['embed_dim']
        property_width = config['property_width']

        bert_config = BertConfig.from_json_file(config['bert_config_text'])
        #self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder_name, config=bert_config)
        self.text_encoder = BertForMaskedLM(config=bert_config)
        text_width = self.text_encoder.config.hidden_size

        self.property_proj = nn.Linear(property_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        self.itm_head = nn.Linear(text_width*2, 2)

        self.property_embed = nn.Linear(1,property_width)
        bert_config2 = BertConfig.from_json_file(config['bert_config_property'])
        self.property_encoder = BertForMaskedLM(config=bert_config2).bert
        self.property_mtr_head=nn.Sequential(nn.Linear(property_width,property_width),
                                             nn.GELU(),
                                             nn.LayerNorm(property_width,bert_config.layer_norm_eps),
                                             nn.Linear(property_width,1))
        self.property_cls = nn.Parameter(torch.zeros(1, 1, property_width))
        self.property_mask = nn.Parameter(torch.zeros(1, 1, property_width))

        # create momentum models
        self.property_encoder_m = BertForMaskedLM(config=bert_config2).bert
        self.property_proj_m = nn.Linear(property_width, embed_dim)
        self.text_encoder_m = BertForMaskedLM(config=bert_config)
        self.text_proj_m = nn.Linear(text_width, embed_dim)

        self.model_pairs = [[self.property_encoder, self.property_encoder_m],
                            [self.property_proj, self.property_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]

        self.copy_params()

        # create the queue
        self.register_buffer("prop_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.prop_queue = nn.functional.normalize(self.prop_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)






    def forward(self, property_original, text_input_ids,text_attention_mask, alpha=0):    #text=(batch*len,batch*len)

        with torch.no_grad():
            self.temp.clamp_(0.01, 0.5)
        
        # property embedding
        property1=self.property_embed(property_original.unsqueeze(2)) #batch*12*feature
        # 50 percent random masking on properties
        property_mask = self.property_mask.expand(property_original.size(0),property_original.size(1), -1)
        mpm_mask=torch.bernoulli(torch.ones_like(property_original)*0.5)#.to(torch.bool)
        mpm_mask_expand=mpm_mask.unsqueeze(2).repeat(1,1,property_mask.size(2))
        property_masked = property1 * (1 - mpm_mask_expand) + property_mask * mpm_mask_expand
        # add CLS token + embedding
        property=torch.cat([self.property_cls.expand(property_original.size(0),-1,-1),property_masked],dim=1)
        prop_embeds = self.property_encoder(inputs_embeds=property,return_dict=True).last_hidden_state   #batch*len(=patch**2+1)*feature

        prop_atts = torch.ones(prop_embeds.size()[:-1], dtype=torch.long).to(property.device)    #batch*len
        prop_feat = F.normalize(self.property_proj(prop_embeds[:, 0, :]), dim=-1)

        text_embeds = self.text_encoder.bert(text_input_ids, attention_mask=text_attention_mask,
                                             return_dict=True, mode='text').last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        # get momentum features

        with torch.no_grad():
            self._momentum_update()
            prop_embeds_m = self.property_encoder_m(inputs_embeds=property,return_dict=True).last_hidden_state
            prop_feat_m = F.normalize(self.property_proj_m(prop_embeds_m[:, 0, :]), dim=-1)
            prop_feat_all = torch.cat([prop_feat_m.t(), self.prop_queue.clone().detach()], dim=1)

            text_embeds_m = self.text_encoder_m.bert(text_input_ids, attention_mask=text_attention_mask,return_dict=True, mode='text').last_hidden_state
            text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            sim_i2t_m = prop_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ prop_feat_all / self.temp
            sim_i2i_m = prop_feat_m @ prop_feat_all / self.temp
            sim_t2t_m = text_feat_m @ text_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(property.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
            sim_i2i_targets = alpha * F.softmax(sim_i2i_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2t_targets = alpha * F.softmax(sim_t2t_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = prop_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ prop_feat_all / self.temp
        sim_i2i = prop_feat @ prop_feat_all / self.temp
        sim_t2t = text_feat @ text_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_i2i_targets, dim=1).mean()
        loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_t2t_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i + loss_i2i + loss_t2t) / 2

        self._dequeue_and_enqueue(prop_feat_m, text_feat_m)

        ###================ ITM =================###
        # forward the positve image-text pair
        pos_pos_prop = self.text_encoder.bert(encoder_embeds = prop_embeds,
                                        attention_mask = prop_atts,
                                        encoder_hidden_states = text_embeds,
                                        encoder_attention_mask = text_attention_mask,
                                        return_dict = True,
                                        mode = 'fusion',
                                       ).last_hidden_state[:, 0, :]
        pos_pos_text_full = self.text_encoder.bert(encoder_embeds = text_embeds,
                                        attention_mask = text_attention_mask,
                                        encoder_hidden_states = prop_embeds,
                                        encoder_attention_mask = prop_atts,
                                        return_dict = True,
                                        mode = 'fusion',
                                       ).last_hidden_state
        pos_pos_text = pos_pos_text_full[:, 0, :]
        pos_pos=torch.cat([pos_pos_prop,pos_pos_text],dim=-1)

        with torch.no_grad():
            bs = property.size(0)
            #hard
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)
            #easy
            #weights_i2t = torch.ones((bs,bs)) / (bs - 1)
            #weights_t2i = torch.ones((bs,bs)) / (bs - 1)

            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        prop_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            prop_embeds_neg.append(prop_embeds[neg_idx])
        prop_embeds_neg = torch.stack(prop_embeds_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text_attention_mask, text_atts_neg], dim=0)
        prop_embeds_all = torch.cat([prop_embeds_neg, prop_embeds], dim=0)
        prop_atts_all = torch.cat([prop_atts, prop_atts], dim=0)

        pos_neg_prop = self.text_encoder.bert(encoder_embeds = prop_embeds_all,
                                        attention_mask = prop_atts_all,
                                        encoder_hidden_states = text_embeds_all,
                                        encoder_attention_mask = text_atts_all,
                                        return_dict = True,
                                        mode = 'fusion',
                                       ).last_hidden_state[:, 0, :]
        pos_neg_text = self.text_encoder.bert(encoder_embeds = text_embeds_all,
                                        attention_mask = text_atts_all,
                                        encoder_hidden_states = prop_embeds_all,
                                        encoder_attention_mask = prop_atts_all,
                                        return_dict = True,
                                        mode = 'fusion',
                                       ).last_hidden_state[:, 0, :]
        pos_neg=torch.cat([pos_neg_prop,pos_neg_text],dim=-1)

        vl_embeddings = torch.cat([pos_pos,pos_neg], dim=0)
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(property.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        ##================= MLM ========================##
        input_ids = text_input_ids.clone()
        labels = input_ids.clone()[:,1:]

        with torch.no_grad():
            #prop_embeds_m = self.property_encoder_m(inputs_embeds=property,attention_mask=prop_atts,return_dict=True).last_hidden_state
            logits_m = self.text_encoder_m(input_ids,
                                           attention_mask = text_attention_mask,
                                           encoder_hidden_states = prop_embeds_m,
                                           encoder_attention_mask = prop_atts,
                                           return_dict = True,
                                           is_decoder = True,
                                           return_logits = True,
                                          )[:,:-1,:] #batch*len*30522

        #prop_embeds = self.property_encoder(inputs_embeds=property,attention_mask=prop_atts,return_dict=True).last_hidden_state
        mlm_output = self.text_encoder(input_ids,
                                       attention_mask = text_attention_mask,
                                       encoder_hidden_states = prop_embeds,
                                       encoder_attention_mask = prop_atts,
                                       return_dict=True,
                                       is_decoder=True,
                                       return_logits=True,
                                      )[:,:-1,:]

        loss_fct = nn.CrossEntropyLoss()
        loss_mlm = loss_fct(mlm_output.permute((0,2,1)), labels)

        loss_distill_text = -torch.sum(F.log_softmax(mlm_output, dim=-1) * F.softmax(logits_m,dim=-1), dim=-1)
        loss_distill_text = loss_distill_text[labels != 0].mean()
        loss_mlm = (1 - alpha) * loss_mlm + alpha * loss_distill_text

        ##================= MPM ========================##
        target=property_original.clone()
        '''
        with torch.no_grad():   #property:batch*(12+1)
            prop_embeds_masked_m = self.property_encoder_m(inputs_embeds=property,is_decoder=True, return_dict=True).last_hidden_state
            prop_output_m = self.text_encoder_m.bert(encoder_embeds = prop_embeds_masked_m,
                                            attention_mask = prop_atts,
                                            encoder_hidden_states = text_embeds_m,
                                            encoder_attention_mask = text_attention_mask,
                                            return_dict = True,
                                            is_decoder=True,
                                            mode = 'fusion',
                                           ).last_hidden_state[:,:-1,:]
            pred_m=self.property_mtr_head(prop_output_m).squeeze()    #batch*12
        '''
        prop_embeds_causal = self.property_encoder(inputs_embeds=property,is_decoder=True,return_dict=True).last_hidden_state
        prop_output = self.text_encoder.bert(encoder_embeds=prop_embeds_causal,
                                                 attention_mask=prop_atts,
                                                 encoder_hidden_states=text_embeds,
                                                 encoder_attention_mask=text_attention_mask,
                                                 return_dict=True,
                                                 is_decoder=True,
                                                 mode='fusion',
                                                 ).last_hidden_state[:, :-1, :]
        pred = self.property_mtr_head(prop_output).squeeze()  # batch*12

        lossfn = nn.MSELoss()
        #target = (1 - alpha) * target + alpha * pred_m
        loss_mpm = lossfn(pred[(1-mpm_mask).to(bool)], target[(1-mpm_mask).to(bool)])



        return loss_mlm, loss_mpm*5, loss_ita, loss_itm


    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, prop_feat, text_feat):
        # gather keys before updating queue
        #image_feats = concat_all_gather(image_feat)
        #text_feats = concat_all_gather(text_feat)
        prop_feats = prop_feat
        text_feats = text_feat

        batch_size = prop_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.prop_queue[:, ptr:ptr + batch_size] = prop_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
