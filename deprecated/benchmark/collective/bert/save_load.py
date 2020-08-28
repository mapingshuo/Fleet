import os
import json
import paddle.fluid as fluid
class Model():
    def __init__(self):
        self.main_program = None
        self.loss = None
        self.startup_program = None
        self.optimizer = None
        self.parameter_list = []
        self.inputs = []
        self.feeder = None
        self.learning_rate = None
        self.generator = fluid.unique_name.UniqueNameGenerator() 


    def save_program(self, main_prog, startup_prog,
                     program_path, input_list,
                     loss, acc_list, generator_info,learning_rate=None):

        if not os.path.exists(program_path):
            os.makedirs(program_path)
        main_program_str = main_prog.desc.serialize_to_string()
        startup_program_str = startup_prog.desc.serialize_to_string()
        params = main_prog.global_block().all_parameters()
        para_info = []
        for pa in params:
            para_info.append(pa.name)
        with open(program_path + '/input_names', 'w') as fout:
            for input in input_list:
                fout.write("%s\n" % input)
        with open(program_path + '/para_info', 'w') as fout:
            for item in params:
                fout.write("%s:%s\n" % (item.name,item.trainable))
        with open(program_path + '/startup_program', "wb") as fout:
            fout.write(startup_program_str)
        with open(program_path + '/main_program', "wb") as fout:
            fout.write(main_program_str)
        with open(program_path + '/loss_name', 'w') as fout:
            fout.write(loss.name)
        if type(learning_rate) == fluid.Variable:
            with open(program_path + '/lr_name', 'w') as fout:
                fout.write(learning_rate.name)
        with open(program_path + '/unique_name_guard', 'w') as fout:
            for id,value in generator_info.iteritems():
                fout.write("%s:%s\n" % (id,value))
        with open(program_path + '/acc', 'w') as fout:
            for acc in acc_list:
                fout.write("%s\n" % acc.name)

    def load_model(self, program_input):
        with open(program_input + '/startup_program', "rb") as fin:
            program_desc_str = fin.read()
            self.startup_program = fluid.Program.parse_from_string(program_desc_str)

        with open(program_input + '/main_program', "rb") as fin:
            program_desc_str = fin.read()
            self.main_program = fluid.Program.parse_from_string(program_desc_str)

        with open(program_input + '/para_info', 'r') as fin:
            for line in fin:
                current_para = line[:-1]
                self.parameter_list.append(current_para)

        input_list = []
        with open(program_input + '/input_names', 'r') as fin:
            for line in fin:
                current_input = line[:-1]
                input_list.append(current_input)

        with open(program_input + '/loss_name', 'r') as fin:
            loss_name = fin.read()

        with open(program_input + '/unique_name_guard', 'r') as fin:
            for line in fin:
                current_guard = line[:-1].split(":")
                self.generator.ids[current_guard[0]] = int(current_guard[1])

        if os.path.exists(program_input + '/lr_name'):
            with open(program_input + '/lr_name', 'r') as fin:
                lr_name = fin.read()
        else:
            lr_name = None

        for item in self.parameter_list:
            para1 = self.startup_program.global_block().var(item)
            para1.regularizer = None
            para1.optimize_attr = {'learning_rate': 1.0}
            para1.trainable = True
            para1.is_distributed = False
	    para2 = self.main_program.global_block().var(item)
            para2.regularizer = None
            para2.optimize_attr = {'learning_rate': 1.0}
            para2.trainable = True
            para2.is_distributed = False
        exe = fluid.Executor(fluid.CPUPlace())
        for var in self.main_program.list_vars():
            if var.name == loss_name:
                self.loss = var
            if var.name in input_list:
                self.inputs.append(var)
            if lr_name != None:
                if var.name == lr_name:
                    self.lr = var
        
