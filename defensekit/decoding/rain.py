"""
RAIN
====
This module contains the implementation of the RAIN algorithm for decoding.
Adapted from https://github.com/SafeAILab/RAIN
TODO: Figure out how to deal with this?
"""

import torch
import numpy as np
import copy
import json
import os
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from sentence_transformers import SentenceTransformer


__all__ = ["rain"]

gamma = 1.0
maxT=50
minT=5
Vt=0.8
maxlen = 2048

with open('f1.txt') as f:
    fschat = f.read()
with open('f2.txt') as f:
    fsred = f.read()
with open('r1.txt') as f:
    redA = f.read()
with open('r2.txt') as f:
    redB = f.read()

encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').cuda()

def getmaxnew(step):
    '''
    return the length of token set
    '''
    if step == 0:
        return 1
    if step == 1:
        return 2
    if step == 2:
        return 4
    return 10

def find_all_indices(text, substring):
    indices = []
    start_index = 0
    while True:
        index = text.find(substring, start_index)
        if index == -1:
            break
        indices.append(index)
        start_index = index + 1
    return indices


@torch.no_grad()
def simg(dicp, orstate, model, tokenizer, maxlen=1280):
    '''
    simulation generation for more accurate self-evaluation
    '''
    state = copy.deepcopy(orstate)
    past_key_values = None
    while 1:
        if len(state) > maxlen:
            break
        tmpstr = tokenizer.decode(state, skip_special_tokens=True)
        if tmpstr[-1] == ',' or tmpstr[-1] == '.' or tmpstr[-1] == '?' or tmpstr[-1] == ':' or tmpstr[
            -1] == ';' or tmpstr[-1] == '\n':
            break
        inds = find_all_indices(tmpstr, 'USER:')
        if len(inds) > 1:
            break
        probs, past_key_values = getp(state, model, dicp, topk=-1, return_past_key_values=True,
                                      past_key_values=past_key_values)
        token = int(torch.multinomial(probs, num_samples=1))
        state.append(token)
        if token == tokenizer.eos_token_id:
            break
    tmpstr = tokenizer.decode(state, skip_special_tokens=True)
    return tmpstr, state

@torch.no_grad()
def getv(getoken, model, tokenizer, dic, dicp, maxlen):
    '''
    score through self-evaluation
    '''
    text, simgstate = simg(dicp, getoken, model, tokenizer, maxlen)
    inds = find_all_indices(text, 'Human:')
    if len(inds) > 1 + 4:
        text = text[:inds[1 + 4]]
    text = text[inds[4]:]
    if text not in dic:
        textA = fsred + '\n\n' + text + '\n' + redA
        textB = fsred + '\n\n' + text + '\n' + redB
        input_ids = tokenizer(textA, return_tensors="pt").input_ids
        outs = model(input_ids.cuda())
        logits = outs.logits
        last_token_logits = logits[0, -1, :]
        prob = F.softmax(last_token_logits.float(), dim=0)
        p_A = prob[29909].item() # prob of 'A'
        p_B = prob[29933].item() # prob of 'B'
        if p_A > p_B:
            A = 1
        else:
            A = 0
        input_ids = tokenizer(textB, return_tensors="pt").input_ids
        outs = model(input_ids.cuda())
        logits = outs.logits
        last_token_logits = logits[0, -1, :]
        prob = F.softmax(last_token_logits.float(), dim=0)
        p_A = prob[29909].item()
        p_B = prob[29933].item()
        if p_B > p_A:
            B = 1
        else:
            B = 0
        v = (A + B) / 2
        v = (v - 0.5) * 2
        dic[text] = v
    else:
        v = dic[text]
    return v, simgstate, len(simgstate) - len(getoken)


@torch.no_grad()
def getp(state, model, dicp, topk=-1, topp=1.0, temperature=1.0, repetition_penalty=0.0, return_last_logits=False,
         return_past_key_values=False, past_key_values=None):
    '''
    query LLM
    '''
    if tuple(state) not in dicp:
        if past_key_values != None:
            input_ids = torch.tensor([[state[-1]]])
            outs = model(input_ids.cuda(), past_key_values=past_key_values)
        else:
            input_ids = torch.tensor([state])
            outs = model(input_ids.cuda())
        logits = outs.logits
        past_key_values = outs.past_key_values
        last_logits = logits[:, -1, :].float().cpu()
        dicp[tuple(state)] = last_logits
    else:
        last_logits = dicp[tuple(state)]
        past_key_values = None

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, topp, topk
    )
    last_token_logits = logits_processor(None, last_logits)[0]
    probs = torch.softmax(last_token_logits, dim=-1)
    if return_last_logits and return_past_key_values:
        return probs, last_logits, past_key_values
    if return_last_logits:
        return probs, last_logits
    if return_past_key_values:
        return probs, past_key_values
    return probs

@torch.no_grad()
def genc(s, model, tokenizer):
    '''
    vanilla autoregression
    '''
    input_ids = tokenizer(s, return_tensors="pt").input_ids
    outs = model.generate(inputs=input_ids.cuda(), max_length=maxlen, use_cache=False)
    outstr = tokenizer.decode(outs[0], skip_special_tokens=True)
    return outstr

@torch.no_grad()
def group_getp(state, model, dicp, topk=10, maxnew=10, temperature=2.0):
    '''
        group query LLM
    '''
    outs = []
    outsset = []
    etmpp = []
    if maxnew == 1:
        p, last_logits = getp(state, model, dicp, topk=topk, return_last_logits=True, temperature=temperature)
        acp = p.cpu().detach().squeeze(0).numpy()
        legal = np.where(acp > 0)[0]
        acp = acp[legal]
        acp = zip(legal, acp)
        for ac, p in acp:
            outs.append(([ac], p))
        return outs, last_logits

    greedytmpstate = copy.deepcopy(state)
    greedytmplog = torch.tensor(0.0)
    greedytmptokens = []
    greedy_past_key_values = None
    for i in range(maxnew):
        greedyprobs, greedy_past_key_values = getp(greedytmpstate, model, dicp, topk=15, return_past_key_values=True,
                                                   past_key_values=greedy_past_key_values,temperature=temperature)
        greedytoken = int(torch.argmax(greedyprobs))
        greedylogp = torch.log(greedyprobs[greedytoken])
        greedytmplog += greedylogp
        greedytmptokens.append(greedytoken)
        greedytmpstate.append(greedytoken)
    outsset.append(greedytmptokens)

    for _ in range(2 * topk - 1):
        tmpstate = copy.deepcopy(state)
        tmplog = torch.tensor(0.0)
        tmptokens = []
        past_key_values = None
        for i in range(maxnew):
            probs, past_key_values = getp(tmpstate, model, dicp, topk=15, return_past_key_values=True,
                                          past_key_values=past_key_values,temperature=temperature)
            token = int(torch.multinomial(probs, num_samples=1))
            logp = torch.log(probs[token])
            tmplog += logp
            tmptokens.append(token)
            tmpstate.append(token)
        if tmptokens not in outsset:
            outsset.append(tmptokens)
            tmpp = torch.exp(tmplog)
            outs.append((tmptokens, tmpp.item()))
            etmpp.append(tmpp.item())
        if len(outs) >= topk - 1:
            break

    greedytmpp = torch.exp(greedytmplog)
    if len(etmpp) > 0:
        etmpp = np.array(etmpp)
        greedytmpp = min(greedytmpp.item(), etmpp.sum())
        greedytmpp = max(greedytmpp, etmpp.max() + etmpp.min())
    else:
        greedytmpp = greedytmpp.item()
    outs = [(greedytmptokens, greedytmpp)] + outs

    return outs

@torch.no_grad()
def search(root, state, model, tokenizer, dic, dicp, maxlen=1024):
    '''
    inner loop
    '''
    state = copy.deepcopy(state)
    cnode = root
    reward = 0
    action = -1

    while not cnode.isleaf():
        addflag = cnode.checkadd()
        if addflag:
            maxnew = getmaxnew(cnode.step)
            agp = group_getp(state, model, dicp, topk=2, maxnew=maxnew)
            cnode.add(agp)
        action, cnode = cnode.select()
        state.extend(action)

    tmpstr = tokenizer.decode(state, skip_special_tokens=True)
    inds = find_all_indices(tmpstr, 'USER:')
    # check whether the generation is finished
    if len(state) > maxlen or action == tokenizer.eos_token_id or len(inds) > 1 or tokenizer.eos_token_id in state:
        v, embeding_token, path_n = getv(state, model, tokenizer, dic, dicp, maxlen)
    else:
        v, embeding_token, path_n = getv(state, model, tokenizer, dic, dicp, maxlen)
        maxnew = getmaxnew(cnode.step)
        if maxnew == 1:
            gp, egp = group_getp(state, model, dicp, topk=10, maxnew=maxnew)
        else:
            gp = group_getp(state, model, dicp, topk=10, maxnew=maxnew)

            egp = copy.deepcopy(gp)
        p = [i[1] for i in gp]
        act = [i[0] for i in gp]
        acp = np.array(p)
        acp = acp / acp.sum()

        if cnode.parent == None:
            acp = 0.75 * acp + 0.25 * np.ones(len(acp)) / len(acp)
            acp = acp / acp.sum()
        acp = zip(act, acp)
        cnode.expand(root=root, ac_p=acp, reward=reward, state=state, logits=egp)
    cnode.backup(v, embeding_token, tokenizer, encoder, path_n=path_n)



def prepare_logits_processor(
        temperature=1.0, repetition_penalty=0.0, top_p=1.0, top_k=-1
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()

    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list

class node():
    def __init__(self, root, parent, prior_p, tokens=None,self_embeding=None,step=0):
        self.step=step
        self.root = root
        self.parent = parent
        self.tokens=tokens
        self.children = {}
        self.n = 0
        self.fn = 0
        self.q = 0
        self.u = 0
        self.reward = 0
        self.p = prior_p
        self.embeding = self_embeding
        self.self_embeding = self_embeding
        self.cosine_similarities = []
        if parent == None:
            self.root = self
            self.maxqn = 1
            self.minqn = -1


    def get_max_n_action(self):
        if not self.children:
            return None

        max_n_value = max(child.n for child in self.children.values())
        max_n_nodes = [(ac, child) for ac, child in self.children.items() if child.n == max_n_value]
        best_ac_node = max(max_n_nodes, key=lambda x: x[1].q)

        return best_ac_node[0]

    def get_max_nq_value(self):
        best_action = self.get_max_n_action()
        if best_action is None:
            return None
        return self.children[best_action].q,self.children[best_action].fn

    def expand(self, root, ac_p, reward,state=None,logits=None):
        self.reward = reward
        self.logits=logits
        self.state=state
        for ac, p in ac_p:
            self.children[tuple(ac)] = node(root=root, parent=self, prior_p=p,tokens=tuple(ac),step=self.step+1)

    def child_embedings_variance(self):
        if self.isleaf():
            return None
        else:
            child_embedings = []
            for child in self.children.values():
                if child.embeding is None:
                    return None
                else:
                    child_embedings.append(child.embeding)

            child_embedings = torch.stack(child_embedings)
            variance = torch.var(child_embedings, dim=0)
            return variance

    def getqu(self):
        cpuct = 2
        if self.n == 0:
            qh = 0
        else:
            qh = (self.q - self.root.minqn) / (self.root.maxqn - self.root.minqn + 1e-5)

        self.u = cpuct * self.p * np.sqrt(self.parent.n) / (1 + self.n)

        return qh + self.u

    def num_children(self):
        return len(self.children)

    def max_child_q(self):
        if self.isleaf():
            return None
        else:
            child_qs = [child.q for child in self.children.values()]
            return max(child_qs)

    def checkadd(self):

        c = next(iter(self.children.values()))
        the_norm = 0.03 * len(c.tokens)
        the_q = 0.0
        nc=self.num_children()
        if nc<20:
            embeding_var=self.child_embedings_variance()
            if embeding_var is not None:
                norm=torch.norm(embeding_var)
                max_q=self.max_child_q()
                if norm<the_norm and max_q<the_q:
                    c=next(iter(self.children.values()))
                    if len(c.tokens)==1:
                        logits_processor = prepare_logits_processor(
                            top_k=self.num_children() + 1
                        )
                        last_token_logits = logits_processor(None, self.logits)[0]
                        probs = torch.softmax(last_token_logits, dim=-1)
                        acp = probs.detach().squeeze(0).numpy()
                        legal = np.where(acp > 0)[0]
                        acp = acp[legal]
                        acp = acp / acp.sum()
                        if self.isroot():
                            acp = 0.75 * acp + 0.25 * np.ones(len(acp)) / len(acp)
                            acp = acp / acp.sum()
                        ac_p = zip(legal, acp)
                        for ac, p in ac_p:
                            if tuple([ac]) not in self.children:
                                self.children[tuple([ac])] = node(root=self.root, parent=self, prior_p=p,
                                                                  tokens=tuple([ac]),
                                                                  step=self.step + 1)
                        return False
                    else:
                        return True
        return False

    def add(self,agp):
        self.logits=agp+self.logits
        gp = self.logits[:self.num_children() + 1]
        p = [i[1] for i in gp]
        act = [i[0] for i in gp]
        acp = np.array(p)
        acp = acp / acp.sum()

        if self.isroot():
            acp = 0.75 * acp + 0.25 * np.ones(len(acp)) / len(acp)
            acp = acp / acp.sum()
        ac_p = zip(act, acp)
        for ac, p in ac_p:
            if tuple(ac) not in self.children:
                self.children[tuple(ac)] = node(root=self.root, parent=self, prior_p=p, tokens=tuple(ac),step=self.step+1)

    def select(self):

        return max(self.children.items(), key=lambda act_node: act_node[1].getqu())

    def backup(self, v, state, tokenizer, encoder, path_n=0):
        sim_gamma = 0.2
        sim_the = 0.95
        g = (gamma) * v + self.reward
        self.n += 1
        self.fn += 1
        if self.parent:
            path_n += len(self.tokens)
            path_state = state[-path_n:]
            path_text = tokenizer.decode(path_state, skip_special_tokens=True)
            path_embeding = encoder.encode(path_text)
            path_embeding = torch.tensor(path_embeding)
            if self.embeding != None:
                self.embeding += (path_embeding - self.embeding) / self.fn
            else:
                self.embeding = path_embeding
            sims = self.calculate_cosine_similarity(path_embeding)
            for similarity, is_self, snode in sims:
                if is_self or similarity < sim_the:
                    continue
                similarity = similarity * sim_gamma
                before_n = snode.n
                snode.n += similarity
                snode.q = (snode.q * before_n + similarity * g) / snode.n

            self.parent.backup(g, state, tokenizer, encoder, path_n)

        self.q += (g - self.q) / self.n
        if not self.isroot():
            self.root.minqn = min(self.root.minqn, self.q)
            self.root.maxqn = max(self.root.maxqn, self.q)



    def calculate_cosine_similarity(self, path_embeding):
        result = []
        siblings = [child for child in self.parent.children.values()]
        for sibling in siblings:
            if sibling.embeding==None:
                similarity=0.0
            else:
                similarity = cosine_similarity(self.embeding.unsqueeze(0).float(),
                                               sibling.embeding.unsqueeze(0).float())
                similarity=similarity.item()
            is_self = (sibling is self)
            result.append((similarity, is_self, sibling))
        return result

    def isleaf(self):
        return self.children == {}

    def isroot(self):
        return self.parent is None

def node2dic(node, state, tokenizer):
    d = {}
    dd = {}
    tmpstr = tokenizer.decode(state, skip_special_tokens=True)
    for act, node in node.children.items():
        actstr = tokenizer.decode(act, skip_special_tokens=True)
        n = node.n
        q = node.q
        dd[actstr] = (n, q)
    d[tmpstr] = dd
    return d

def save_dict(dict_input, filename):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            dict_existing = json.load(file)
        dict_merged = {**dict_existing, **dict_input}
    else:
        dict_merged = dict_input

    with open(filename, 'w') as file:
        json.dump(dict_merged, file)

@torch.no_grad()
def rain(query, model, tokenizer):
    """
    RAIN is a decoding algorithm.

    :param str query: The input prompt.
    :param Any model: The model.
    :param Any tokenizer: The tokenizer.
    :return str: The generated output.
    """
    model.eval()

    '''
    outer loop
    '''
    dic, dicp = {}, {}

    input_ids = tokenizer(query, return_tensors="pt").input_ids
    slen = input_ids.shape[1]
    state = input_ids.tolist()[0]


    root = node(root=None, parent=None, prior_p=0, step=0)

    initi = 0
    while 1:
        for i in range(initi, max(maxT, initi + 15)):
            search(root, state, model, tokenizer, dic, dicp, maxlen=maxlen)
            try:
                bq, bfn = root.get_max_nq_value()
            except:
                bq, bfn = 0, 0
            if bfn > minT and bq > Vt:
                break
        act_visits = [(act, node.n) for act, node in root.children.items()]
        try:
            acts, visits = zip(*act_visits)
            visits = np.array(visits)
            targetact_probs = (visits) / (visits.sum())
            visits = visits
            act_probs = (visits) / (visits.sum())
            move = acts[int(torch.tensor(act_probs).max(dim=0).indices)]
            move = root.get_max_n_action()
            rootd = node2dic(root, state, tokenizer)

            # args
            index = 1
            outdir = "ourdir"
            save_dict(rootd, '{}_dicv/res_root_{}.json'.format(outdir, index))

            state.extend(move)
            oroot = root
            root = root.children[move]
            root.parent = None
            root.minqn = oroot.minqn
            root.maxqn = oroot.maxqn
            cp = [root.children[i].p for i in root.children]
            cp = np.array(cp)
            cp = 0.75 * cp + 0.25 * np.ones(len(cp)) / len(cp)
            cp = cp / cp.sum()

            for id, i in enumerate(root.children):
                root.children[i].p = cp[id]
            initi = root.fn
        except:
            move = tokenizer.eos_token_id

        tmpstr = tokenizer.decode(state, skip_special_tokens=True)
        inds = find_all_indices(tmpstr, 'USER:')
        if len(inds) > 1:
            break
        if len(state) > maxlen:
            break
        if tokenizer.eos_token_id in state:
            break
        if move == tokenizer.eos_token_id:
            break

    raina = tokenizer.decode(state, skip_special_tokens=True)
    inds = find_all_indices(raina, 'USER:')
    if len(inds) > 1:
        raina = raina[:inds[1]]
    raina = raina[inds[0]:]


    pa = genc(query, model, tokenizer)
    inds = find_all_indices(pa, 'USER:')
    if len(inds) > 1:
        pa = pa[:inds[1]]
    pa = pa[inds[0]:]


    tmp = {'question': query, 'raina': raina, 'pa': pa}
    save_dict(dic, '{}_dicv/res_{}.json'.format(outdir, index))
    return tmp