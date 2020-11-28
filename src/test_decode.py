# encoding: utf-8
import torch
from torch.autograd import Variable
import numpy as np
import json
import pickle  # as pickle
import os
import sys

from utils.loss import NLLLoss
from utils.equ_tools import *
import pdb


from utils import DataLoader
from train import SupervisedTrainer
from model import EncoderRNN, DecoderRNN_1, DecoderRNN_2, DecoderRNN_3, Seq2seq
from utils import NLLLoss, Optimizer, Checkpoint
from utils.config import get_args
from utils.data_tools import *
from utils.equ_tools import *

print_flag_ = 0
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def step_one_test():
    args = get_args()

    print(args)
    one_quest_test_list = read_data_json("./data/test.json")
    print(one_quest_test_list)
    #Checkpoint.CHECKPOINT_DIR_NAME = "0120_0030"
    Checkpoint.CHECKPOINT_DIR_NAME = args.checkpoint_dir_name
    checkpoint_path = os.path.join("./experiment", Checkpoint.CHECKPOINT_DIR_NAME, "best")
    print('-----',args.checkpoint_dir_name, checkpoint_path)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    if args.cuda_use:
        seq2seq = seq2seq.cuda()
    data_loader = DataLoader(args)
    seq2seq.eval()
    evaluator = Evaluator(vocab_dict = data_loader.vocab_dict,
                          vocab_list = data_loader.vocab_list,
                          decode_classes_dict = data_loader.decode_classes_dict,
                          decode_classes_list = data_loader.decode_classes_list,
                          loss = NLLLoss(),
                          cuda_use = args.cuda_use)
    name = args.run_flag
    test_temp_acc, test_ans_acc = evaluator.evaluate(model = seq2seq,
                                                 data_loader = data_loader,
                                                 data_list = one_quest_test_list,
                                                 template_flag = False, # todo: 这里改成了false
                                                 batch_size = 64,
                                                 evaluate_type = 0,
                                                 use_rule = False,
                                                 mode = args.mode,
                                                 post_flag=args.post_flag,
                                                 name_save = name)
    print (test_temp_acc, test_ans_acc)
class Evaluator(object):
    def __init__(self, vocab_dict, vocab_list, decode_classes_dict, decode_classes_list, \
                 loss=NLLLoss(), cuda_use=False):
        self.loss = loss
        self.cuda_use = cuda_use
        if self.cuda_use:
            self.loss.cuda()

        self.vocab_dict = vocab_dict
        self.vocab_list = vocab_list
        self.decode_classes_dict = decode_classes_dict
        self.decode_classes_list = decode_classes_list

        self.pad_in_classes_idx = self.decode_classes_dict['PAD_token']
        self.end_in_classes_idx = self.decode_classes_dict['END_token']

    def _convert_f_e_2_d_sybmbol(self, target_variable):
        new_variable = []
        batch, colums = target_variable.size()
        for i in range(batch):
            tmp = [0] * colums
            for j in range(colums):
                # print target_variable[i][j].data[0]
                try:
                    idx = self.decode_classes_dict[self.vocab_list[target_variable[i][j].item()]]
                    # tmp.append(idx)
                    tmp[j] = idx
                except:
                    pass
                    # print self.vocab_list[target_variable[i][j].data[0]].encode('utf-8')
                # tmp.append(idx)
            new_variable.append(tmp)
        try:
            return Variable(torch.LongTensor(np.array(new_variable)))
            # print ('-------',new_variable)
        except:
            print('-+++++++', new_variable)

    def inverse_temp_to_num(self, equ_list, num_list):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        new_equ_list = []
        for elem in equ_list:
            if 'temp' in elem:
                index = alphabet.index(elem[-1])
                new_equ_list.append(num_list[index])
            elif 'PI' == elem:
                new_equ_list.append('3.14')
            else:
                new_equ_list.append(elem)
        return new_equ_list

    def inverse_temp_to_num_(self, equ_list, num_list):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        new_equ_list = []
        for elem in equ_list:
            if 'temp' in elem:
                index = alphabet.index(elem[-1])
                try:
                    new_equ_list.append(str(num_list[index]))
                except:
                    return []
            elif 'PI' == elem:
                new_equ_list.append('3.14')
            else:
                new_equ_list.append(elem)
        return new_equ_list

    def get_new_tempalte(self, seq_var, num_list):
        equ_list = []
        for idx in seq_var.data.cpu().numpy():
            if idx == self.pad_in_classes_idx:
                break
            equ_list.append(self.decode_classes_list[idx])
        equ_list = self.inverse_temp_to_num_(equ_list, num_list)
        try:
            equ_list = equ_list[:equ_list.index('END_token')]
        except:
            pass
        return equ_list

    def compute_gen_ans(self, seq_var, num_list, post_flag):
        equ_list = []
        for idx in seq_var.data.cpu().numpy():
            if idx == self.pad_in_classes_idx:
                break
            equ_list.append(self.decode_classes_list[idx])
        # equ_list = equ_list[2:]
        try:
            # print equ_list[:equ_list.index('END_token')]
            # print equ_list, num_list,
            equ_list = self.inverse_temp_to_num(equ_list, num_list)
        except:
            # print "inverse error",
            return 'inverse error'
        try:
            equ_list = equ_list[:equ_list.index('END_token')]
            # print num_list
            # print ' '.join(equ_list),
            # if '1' in equ_list:
            #    print 'g1'
            # else:
            #    print
        except:
            # print equ_list
            pass
        # equ_string = ''.join(equ_list)
        # equ_list = split_equation(equ_string)
        if 0:
            print(num_list)
            print(equ_list)
        try:
            # ans = solve_equation(equ_list)
            if print_flag_:
                print('---debb-', num_list)
                print('---debb-', equ_list)
            if post_flag == False:
                ans = solve_equation(equ_list)
            else:
                ans = post_solver(equ_list)
            float(ans)
            return ans
        except:
            if 0:
                print("compute error", post_flag)
            return ("compute error")

    def evaluate(self, model, data_loader, data_list, template_flag, batch_size, \
                 evaluate_type, use_rule, mode, post_flag=False, name_save='train'):
        batch_generator = data_loader.get_batch(data_list, batch_size, template_flag)
        total_num = len(data_list)

        print('total_num:', total_num)
        # if evaluate_type == 0:
        #     teacher_forcing_ratio = 0.0
        # else:
        #     teacher_forcing_ratio = 1.0
        teacher_forcing_ratio = 0
        count = 0
        acc_right = 0

        id_right_and_error = {1: [], 0: [], -1: []}
        id_template = {}
        pg_total_list = []
        xxx = 0

        for batch_data_dict in batch_generator:
            input_variables = batch_data_dict['batch_encode_pad_idx']
            input_lengths = batch_data_dict['batch_encode_len']

            input_variables = Variable(torch.LongTensor(input_variables))
            if self.cuda_use:
                input_variables = input_variables.cuda()

            batch_index = batch_data_dict['batch_index']
            batch_text = batch_data_dict['batch_text']
            batch_num_list = batch_data_dict['batch_num_list']
            batch_solution = batch_data_dict['batch_solution']
            batch_size = len(batch_index)

            if template_flag == True:
                target_variables = batch_data_dict['batch_decode_pad_idx']
                target_lengths = batch_data_dict['batch_decode_len']

                target_variables = Variable(torch.LongTensor(target_variables))
                if self.cuda_use:
                    target_variables = target_variables.cuda()
            else:
                target_variables = None

            decoder_outputs, decoder_hidden, symbols_list = \
                model(input_variable=input_variables,
                      input_lengths=input_lengths,
                      target_variable=target_variables,
                      template_flag=template_flag,
                      teacher_forcing_ratio=teacher_forcing_ratio,
                      mode=mode,
                      use_rule=use_rule,
                      use_cuda=self.cuda_use,
                      vocab_dict=self.vocab_dict,
                      vocab_list=self.vocab_list,
                      class_dict=self.decode_classes_dict,
                      class_list=self.decode_classes_list)

            seqlist = symbols_list
            seq_var = torch.cat(seqlist, 1)
            batch_pg = []
            # batch_att = []
            for i in range(batch_size):
                wp_index = batch_index[i]
                p_list = []
                # att_tmp_list = []
                for j in range(len(decoder_outputs)):
                    # mm_elem_idx = seq_var[i][j].cpu().data.numpy()[0]
                    mm_elem_idx = seq_var[i][j].cpu().data.numpy().tolist()
                    # cur_att = att_list[j][i,0,:].cpu().data.numpy().tolist()
                    # print mm_elem_idx, self.end_in_classes_idx, mm_elem_idx == self.end_in_classes_idx
                    if mm_elem_idx == self.end_in_classes_idx:
                        break
                    num_p = decoder_outputs[j][i].topk(1)[0].cpu().data.numpy()[0]
                    p_list.append(str(num_p))
                    # att_tmp_list.append(cur_att)
                batch_pg.append(p_list)
                # batch_att.append(att_tmp_list)

            # xxx += batch_size
            # print '>>>>>', xxx

            for i in range(batch_size):
                if 0:  # print_flag_:
                    print('index', batch_index[i], '--', )
                    print('text', batch_text[i].encode('utf-8'))
                if template_flag == True:
                    target_equ = target_variables[i].cpu().data.numpy()
                    tmp_equ = []
                    if print_flag_:
                        print(target_equ)
                    for target_idx in target_equ:
                        # elem = self.decode_classes_list[target_idx]
                        elem = self.vocab_list[target_idx]
                        if elem == 'END_token':
                            break
                        tmp_equ.append(elem)
                # print ' '.join(tmp_equ),
                # if 0 and '1' in tmp_equ:
                #    print 't1'
                # else:
                #    print
                # print target_equ
                gen_ans = self.compute_gen_ans(seq_var[i], batch_num_list[i], post_flag)
                gen_equ = self.get_new_tempalte(seq_var[i], batch_num_list[i])
                # print('gen_ans:', gen_ans, 'gen_equ:', gen_equ)
                target_ans = batch_solution[i]
                # print gen_equ
                # print 'gen_ans', gen_ans, '--', 'target_ans', target_ans, '---',
                # pdb.set_trace()
                pg_total_list.append(dict({'index': batch_index[i], 'gen_equ': gen_equ, \
                                           'pg': batch_pg[i], 'gen_ans': gen_ans, 'ans': target_ans}))
                if 'error' in gen_ans:
                    # print False
                    # print
                    id_right_and_error[-1].append(batch_index[i])
                    continue
                else:
                    # print 'kkkkk', gen_ans, target_ans
                    # if '%' in str(gen_ans):
                    #    gen_ans = str(gen_ans)[:-1]
                    # if '%' in str(target_ans):
                    #    target_ans = str(target_ans)[:-1]
                    if abs(float(gen_ans) - float(target_ans)) < 1e-5:
                        acc_right += 1
                        id_right_and_error[1].append(batch_index[i])
                        if 0:  # print_flag_:
                            print(True)
                    else:
                        id_right_and_error[0].append(batch_index[i])
                        if 0:  # print_flag_:
                            print(False)
                            print()
                        # print
                        continue
                if 0:  # print_flag_:
                    print()

            if template_flag == True:
                target_variables = self._convert_f_e_2_d_sybmbol(target_variables)
                if self.cuda_use:
                    target_variables = target_variables.cuda()
                for i in range(batch_size):
                    right_flag = 0
                    for j in range(target_variables.size(1)):
                        if seq_var[i][j].item() == self.end_in_classes_idx and \
                                target_variables[i][j].item() == self.end_in_classes_idx:
                            right_flag = 1
                            break
                        if target_variables[i][j].item() != seq_var[i][j].item():
                            break
                    count += right_flag

        # with open("./data/id_right_and_error.json", 'w') as f:
        #    json.dump(id_right_and_error, f)
        # with open("./data/id_template.json", 'w') as f:
        #    json.dump(id_template, f)
        # pdb.set_trace()
        with open("./data/pg_seq_norm_" + str(post_flag) + "_" + name_save + ".json", 'w') as f:
            json.dump(pg_total_list, f)
        if template_flag == True:
            print('--------', acc_right, total_num)
            return count * 1.0 / total_num, acc_right * 1.0 / total_num, pg_total_list
        else:
            print('--------', acc_right, total_num)
            return 0, acc_right * 1.0 / total_num, pg_total_list

    def _init_rl_state(self, encoder_hidden, bi_flag):
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h, bi_flag) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden, bi_flag)
        return encoder_hidden

    def _cat_directions(self, h, bi_flag):
        if bi_flag:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        h = torch.cat(h[0:h.size(0)], 1)
        return h

    def _write_data_json(self, data, filename):
        with open(filename, 'wb') as f:
            json.dump(data, f)

    def _write_data_pickle(self, data, filename):
        with open(filename, 'w') as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    step_one_test()
