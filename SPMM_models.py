from xbert import BertConfig, BertForMaskedLM
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed
import pytorch_lightning as pl
from scheduler import create_scheduler


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class SPMM(pl.LightningModule):
    def __init__(self, tokenizer=None, config=None, loader_len=0, no_train=False):
        super().__init__()
        self.automatic_optimization = False
        self.config = config
        self.tokenizer = tokenizer
        self.training_step_outputs = []

        embed_dim = config['embed_dim']

        bert_config = BertConfig.from_json_file(config['bert_config_text'])
        self.text_encoder = BertForMaskedLM(config=bert_config)
        text_width = self.text_encoder.config.hidden_size
        property_width = text_width

        self.property_proj = nn.Linear(property_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width * 2, 2)

        self.property_embed = nn.Linear(1, property_width)
        bert_config2 = BertConfig.from_json_file(config['bert_config_property'])
        self.property_encoder = BertForMaskedLM(config=bert_config2).bert
        self.property_mtr_head = nn.Sequential(nn.Linear(property_width, property_width),
                                               nn.GELU(),
                                               nn.LayerNorm(property_width, bert_config.layer_norm_eps),
                                               nn.Linear(property_width, 1))
        self.property_cls = nn.Parameter(torch.zeros(1, 1, property_width))
        self.property_mask = nn.Parameter(torch.zeros(1, 1, property_width))    # unk token for PV

        # create momentum models
        self.property_encoder_m = BertForMaskedLM(config=bert_config2).bert
        self.property_proj_m = nn.Linear(property_width, embed_dim)
        self.text_encoder_m = BertForMaskedLM(config=bert_config)
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        for p in self.property_encoder_m.parameters():  p.requires_grad = False
        for p in self.property_proj_m.parameters():     p.requires_grad = False
        for p in self.text_encoder_m.parameters():      p.requires_grad = False
        for p in self.text_proj_m.parameters():         p.requires_grad = False

        self.model_pairs = [[self.property_encoder, self.property_encoder_m],
                            [self.property_proj, self.property_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]

        self.copy_params()

        # create the queue
        if not no_train:
            self.temp = nn.Parameter(torch.ones([]) * config['temp'])
            self.mlm_probability = config['mlm_probability']
            self.warmup_steps = config['schedular']['warmup_epochs']
            self.loader_len = loader_len
            self.momentum = config['momentum']
            self.queue_size = config['queue_size']
            self.register_buffer("prop_queue", torch.randn(embed_dim, self.queue_size))
            self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            self.prop_queue = nn.functional.normalize(self.prop_queue, dim=0)
            self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def forward(self, property_original, text_input_ids, text_attention_mask, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.01, 0.5)
        property_feature = self.property_embed(property_original.unsqueeze(2))

        unk_tokens = self.property_mask.expand(property_original.size(0), property_original.size(1), -1)
        mpm_mask = torch.bernoulli(torch.ones_like(property_original) * 0.5)    # 1 for mask, 0 for keep
        mpm_mask_expand = mpm_mask.unsqueeze(2).repeat(1, 1, unk_tokens.size(2))
        property_masked = property_feature * (1 - mpm_mask_expand) + unk_tokens * mpm_mask_expand
        properties = torch.cat([self.property_cls.expand(property_original.size(0), -1, -1), property_masked], dim=1)

        prop_embeds = self.property_encoder(inputs_embeds=properties, return_dict=True).last_hidden_state
        prop_atts = torch.ones(prop_embeds.size()[:-1], dtype=torch.long).to(properties.device)
        prop_feat = F.normalize(self.property_proj(prop_embeds[:, 0, :]), dim=-1)

        text_embeds = self.text_encoder.bert(text_input_ids, attention_mask=text_attention_mask, return_dict=True, mode='text').last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        # get momentum features

        with torch.no_grad():
            self._momentum_update()
            prop_embeds_m = self.property_encoder_m(inputs_embeds=properties, return_dict=True).last_hidden_state
            prop_feat_m = F.normalize(self.property_proj_m(prop_embeds_m[:, 0, :]), dim=-1)
            prop_feat_all = torch.cat([prop_feat_m.t(), self.prop_queue.clone().detach()], dim=1)

            text_embeds_m = self.text_encoder_m.bert(text_input_ids, attention_mask=text_attention_mask, return_dict=True, mode='text').last_hidden_state
            text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            sim_i2t_m = prop_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ prop_feat_all / self.temp
            sim_i2i_m = prop_feat_m @ prop_feat_all / self.temp
            sim_t2t_m = text_feat_m @ text_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(properties.device)
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
        if torch.isnan(sim_i2t).any() or torch.isnan(sim_t2i).any() or torch.isnan(loss_ita):
            return torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)

        # ================ ITM ================= #
        # forward the positve image-text pair
        pos_pos_prop = self.text_encoder.bert(encoder_embeds=prop_embeds,
                                              attention_mask=prop_atts,
                                              encoder_hidden_states=text_embeds,
                                              encoder_attention_mask=text_attention_mask,
                                              return_dict=True,
                                              mode='fusion',
                                              ).last_hidden_state[:, 0, :]
        pos_pos_text_full = self.text_encoder.bert(encoder_embeds=text_embeds,
                                                   attention_mask=text_attention_mask,
                                                   encoder_hidden_states=prop_embeds,
                                                   encoder_attention_mask=prop_atts,
                                                   return_dict=True,
                                                   mode='fusion',
                                                   ).last_hidden_state
        pos_pos_text = pos_pos_text_full[:, 0, :]
        pos_pos = torch.cat([pos_pos_prop, pos_pos_text], dim=-1)

        with torch.no_grad():
            bs = properties.size(0)
            # hard
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)

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

        pos_neg_prop = self.text_encoder.bert(encoder_embeds=prop_embeds_all,
                                              attention_mask=prop_atts_all,
                                              encoder_hidden_states=text_embeds_all,
                                              encoder_attention_mask=text_atts_all,
                                              return_dict=True,
                                              mode='fusion',
                                              ).last_hidden_state[:, 0, :]
        pos_neg_text = self.text_encoder.bert(encoder_embeds=text_embeds_all,
                                              attention_mask=text_atts_all,
                                              encoder_hidden_states=prop_embeds_all,
                                              encoder_attention_mask=prop_atts_all,
                                              return_dict=True,
                                              mode='fusion',
                                              ).last_hidden_state[:, 0, :]
        pos_neg = torch.cat([pos_neg_prop, pos_neg_text], dim=-1)

        vl_embeddings = torch.cat([pos_pos, pos_neg], dim=0)
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(properties.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        self._dequeue_and_enqueue(prop_feat_m, text_feat_m)

        # ================= MLM ================= #
        input_ids = text_input_ids.clone()
        labels = input_ids.clone()[:, 1:]

        with torch.no_grad():
            logits_m = self.text_encoder_m(input_ids,
                                           attention_mask=text_attention_mask,
                                           encoder_hidden_states=prop_embeds_m,
                                           encoder_attention_mask=prop_atts,
                                           return_dict=True,
                                           is_decoder=True,
                                           return_logits=True,
                                           )[:, :-1, :]

        mlm_output = self.text_encoder(input_ids,
                                       attention_mask=text_attention_mask,
                                       encoder_hidden_states=prop_embeds,
                                       encoder_attention_mask=prop_atts,
                                       return_dict=True,
                                       is_decoder=True,
                                       return_logits=True,
                                       )[:, :-1, :]

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss_mlm = loss_fct(mlm_output.permute((0, 2, 1)), labels)

        loss_distill_text = -torch.sum(F.log_softmax(mlm_output, dim=-1) * F.softmax(logits_m, dim=-1), dim=-1)
        loss_distill_text = loss_distill_text[labels != 0].mean()
        loss_mlm = (1 - alpha) * loss_mlm + alpha * loss_distill_text

        # ================= MPM ================= #
        target = property_original.clone()
        prop_embeds_causal = self.property_encoder(inputs_embeds=properties, is_decoder=True, return_dict=True).last_hidden_state
        prop_output = self.text_encoder.bert(encoder_embeds=prop_embeds_causal,
                                             attention_mask=prop_atts,
                                             encoder_hidden_states=text_embeds,
                                             encoder_attention_mask=text_attention_mask,
                                             return_dict=True,
                                             is_decoder=True,
                                             mode='fusion',
                                             ).last_hidden_state[:, :-1, :]
        pred = self.property_mtr_head(prop_output).squeeze()

        lossfn = nn.MSELoss()
        loss_mpm = lossfn(pred[(1 - mpm_mask).to(bool)], target[(1 - mpm_mask).to(bool)])

        return loss_mlm, loss_mpm * 5, loss_ita, loss_itm

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
    def _dequeue_and_enqueue(self, img_feat, text_feat):
        img_feats = concat_all_gather(img_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = img_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.prop_queue[:, ptr:ptr + batch_size] = img_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def mask_pv(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
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

    def configure_optimizers(self):
        arg_opt = self.config['optimizer']
        optimizer = torch.optim.AdamW(self.parameters(), lr=arg_opt['lr'], weight_decay=arg_opt['weight_decay'])
        arg_sche = AttrDict(self.config['schedular'])
        scheduler, _ = create_scheduler(arg_sche, optimizer)
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        print('qqq', metric)

    def training_step(self, train_batch, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad()
        prop, text = train_batch
        text_input = self.tokenizer(text, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(prop.device)
        # print(text_input.input_ids[:4], prop[:4], text_input.input_ids.shape)
        alpha = self.config['alpha'] if self.current_epoch > 0 else self.config['alpha'] * min(1., batch_idx / self.loader_len)

        loss_mlm, loss_mpm, loss_ita, loss_itm = self(prop, text_input.input_ids[:, 1:], text_input.attention_mask[:, 1:], alpha=alpha)
        loss = loss_mlm + loss_mpm + loss_ita + loss_itm
        if loss != torch.tensor(0.):
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.)
            optimizer.step()
        else:
            print('aaaaaaaaaaaa')
        if self.global_rank == 0:
            self.log('lr', optimizer.param_groups[0]["lr"], prog_bar=True)
            self.log('loss_mlm', loss_mlm, prog_bar=True)
            self.log('loss_mpm', loss_mpm, prog_bar=True)
            self.log('loss_ita', loss_ita, prog_bar=True)
            self.log('loss_itm', loss_itm, prog_bar=True)

        step_size = 100
        warmup_iterations = self.warmup_steps * step_size
        if self.current_epoch > 0 and batch_idx == 0:
            scheduler.step(self.current_epoch + self.warmup_steps)
        else:
            if self.current_epoch == 0 and batch_idx % step_size == 0 and batch_idx <= warmup_iterations:
                scheduler.step(batch_idx // step_size)
        self.training_step_outputs.append(torch.tensor([loss_mlm, loss_mpm, loss_ita, loss_itm]))
        return torch.tensor([loss_mlm, loss_mpm, loss_ita, loss_itm])

    def on_train_epoch_end(self):    # outputs: collection of returns from 'training_step'
        tmp = torch.stack(self.training_step_outputs[-1000:]).mean(dim=0).tolist()
        if self.global_rank == 0:
            print(f'\n mean loss: {tmp[0]:.4f}, {tmp[1]:.4f}, {tmp[2]:.4f}, {tmp[3]:.4f}')
        self.training_step_outputs.clear()


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
