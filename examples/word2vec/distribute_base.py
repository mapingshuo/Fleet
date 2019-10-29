#!/usr/bin/python
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import sys
import os
import re
import io
import commands
import logging
import time
import numpy as np
import thread
import paddle
import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.incubate.fleet.utils import fleet_barrier_util
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig
from dataset_generator import prepare_data

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


class FleetDistRunnerBase(object):
    """
    Distribute training base class:
        This class abstracts the training process into several major steps:
        1. input_data: input data of network, this function should be realized by user
        2. net: network definition, this function should be defined by user
        3. infer net: network for test, this function should be defined by user
        4. run_pserver: run parameter server program
        5. run_trainer: run trainer, choose the way of training network according to requirement params
        6. run_infer: prediction based on the trained model
        7. run_local: run local program
        8. run_local_cluster: using multiprocess simulation distributed training
        9. run_cloud: distributed training using parameter server
        10. dataset_reader: using dataset method get data, this function should be realized by user
        11. runtime_main: program entry, get the environment parameters, decide which function to call
    """

    def input_data(self, params):
        """
        Function input_data: Definition of input data format in the network
        Args:
            :params: the hyper parameters of network
        Returns:
            defined by users
        """
        raise NotImplementedError(
            "input_data should be implemented by child classes.")

    def net(self, inputs, params):
        """
        Function net: Definition of network structure
        Args:
            :inputs: input data, eg: dataset and labels. defined by funtion: self.input_data
            :params: the hyper parameters of network
        Returns:
            evaluation parameter, defined by users
        """
        raise NotImplementedError(
            "net should be implemented by child classes.")

    def infer_net(self, params):
        """
        Function net: Definition of infer network structure, This function is not required
                      if the predict is same with the training logic
        Args:
            :params: the hyper parameters of network
        Returns:
            evaluation parameter, defined by users
         """
        raise NotImplementedError(
            "net should be implemented by child classes.")

    def dataset_reader(self, inputs, params):
        """
        Function dataset_reader: define the data read method by fluid.dataset.DatasetFactory
        help: https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/api_cn/dataset_cn.html#fluid-dataset
        Args:
           :params inputs: input data, eg: dataset and labels. defined by funtion: self.input_data
           :params params: the hyper parameters of network
        Returns:
           defined by user
        """
        raise NotImplementedError(
            "dataset_reader should be implemented by child classes.")

    def runtime_main(self, params):
        """
        Function runtime_main: the entry point for program running
        Args:
            :params params: the hyper parameters of network
        """
        if params.training_method == "local":
            logger.info("Local train start")
            self.run_local(params)
        elif params.training_method == "local_cluster":
            logger.info("Local cluster train start")
            self.run_local_cluster(params)
        elif params.training_method == "cloud":
            logger.info("Cloud train start")
            self.run_cloud(params)

    def run_cloud(self, params):
        # Step1: get the environment variable
        params.cpu_num = int(os.getenv("CPU_NUM"))

        # Step2: Init distribute training role
        self.role = role_maker.PaddleCloudRoleMaker()
        fleet.init(self.role)

        # Step3: decide distribute training strategy between PSERVER & TRAINER
        self.strategy = DistributeTranspilerConfig()
        self.strategy.sync_mode = False
        self.strategy.geo_sgd_mode = True
        self.strategy.geo_sgd_need_push_nums = 400
        # update decay step when using GEO-SGD
        params.decay_steps = params.decay_steps/self.role.worker_num()

        # step4: Creat network and minimize loss
        self.inputs = self.input_data(params)
        self.loss = self.net(self.inputs, params)
        self.optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.layers.exponential_decay(
                learning_rate=params.learning_rate,
                decay_steps=params.decay_steps,
                decay_rate=params.decay_rate,
                staircase=True))
        self.optimizer = fleet.distributed_optimizer(
            self.optimizer, self.strategy)
        self.optimizer.minimize(self.loss)

        # Step5: According to the parameters-> TRAINING_ROLE, decide which method to run
        if self.role.is_server():
            self.run_pserver(params)
        elif self.role.is_worker():
            self.run_dataset_trainer(params)
        else:
            raise ValueError(
                "Please choice training role for current node : PSERVER / TRAINER")

        logger.info("Distribute train success!")

    def run_pserver(self, params):
        """
        Function run_pserver: Operation method of parameter server
        Args
            :params the hyper parameters of network
        Returns:
            None
        """
        fleet.init_server()
        logger.info("PServer init success!")
        fleet.run_server()

    def run_dataset_trainer(self, params):
        """
        Function run_dataset_trainer: Operation method of training node
        Args:
            :params params: the hyper parameters of network
        Returns
            :train_result: the dict of training log
        """
        # step5: define Executor and run startup program
        fleet.init_worker()
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fleet.startup_program)

        # step6: init dataset reader
        # Notice: Both dataset and py_reader method don't using feed={dict} to input data
        # Paddle Fluid enter data by variable name
        # When we do the definition of the reader, the program has established the workflow
        dataset = self.dataset_reader(self.inputs, params)
        file_list = [str(params.train_files_path) + "/%s" % x
                     for x in os.listdir(params.train_files_path)]
        if params.training_method == "local_cluster":
            file_list = fleet.split_files(file_list)
        dataset.set_filelist(file_list)
        logger.info("file list: {}".format(file_list))

        all_examples = self.get_example_num(file_list)

        # step7: begin to train your model, good luck
        for epoch in range(params.epochs):
            class fetch_vars(fluid.executor.FetchHandler):
                def handler(self, fetch_target_vars):
                    loss_value = fetch_target_vars[0]
                    logger.info(
                        "epoch -> {}, loss -> {}, at: {}".format(epoch, loss_value, time.ctime()))

            start_time = time.time()
            # Notice: function train_from_dataset does not return fetch value
            # Using fetch_vars to get more information
            exe.train_from_dataset(program=fleet.main_program, dataset=dataset,
                                   fetch_handler=fetch_vars([self.loss.name], 5, True))
            end_time = time.time()
            training_time = float(end_time - start_time)
            speed = float(all_examples) / training_time
            logger.info("epoch: %d finished, using time: %f s ,speed: %f sentences/s" %
                        (epoch, training_time, speed))

            if self.role.is_first_worker() and params.test:
                model_path = str(params.model_path) + '/trainer_' + \
                    str(self.role.worker_index()) + '_epoch_' + str(epoch)
                fleet.save_persistables(executor=exe, dirname=model_path)
                fleet.save_inference_model

        logger.info("Train Success!")
        fleet.stop_worker()

    def run_infer(self, params, model_path):
        """
        Function run_infer: Operation method of prediction
        Args:
            :params params: the hyper parameters of network
        Returns
            :infer_result, type:dict, record the evalution parameter and program resource usage situation
        """
        params.vocab_size, test_reader, id2word = prepare_data(
            params.test_files_path, params.infer_dict_path, batch_size=params.infer_batch_size)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        infer_result = {}
        startup_program = fluid.framework.Program()
        test_program = fluid.framework.Program()

        with fluid.framework.program_guard(test_program, startup_program):
            values, pred = self.infer_net(params)
            logger.info("model_path: %s" % model_path)
            fluid.io.load_persistables(
                executor=exe, dirname=model_path, main_program=fluid.default_main_program())

            accum_num = 0
            accum_num_sum = 0.0
            step_id = 0
            for data in test_reader():
                step_id += 1
                b_size = len([dat[0] for dat in data])
                wa = np.array([dat[0] for dat in data]).astype(
                    "int64").reshape(b_size, 1)
                wb = np.array([dat[1] for dat in data]).astype(
                    "int64").reshape(b_size, 1)
                wc = np.array([dat[2] for dat in data]).astype(
                    "int64").reshape(b_size, 1)

                label = [dat[3] for dat in data]
                input_word = [dat[4] for dat in data]
                para = exe.run(fluid.default_main_program(),
                               feed={
                                   "analogy_a": wa, "analogy_b": wb, "analogy_c": wc,
                                   "all_label":
                                       np.arange(params.vocab_size).reshape(
                                           params.vocab_size, 1).astype("int64"),
                },
                    fetch_list=[pred.name, values],
                    return_numpy=False)
                pre = np.array(para[0])
                val = np.array(para[1])
                for ii in range(len(label)):
                    top4 = pre[ii]
                    accum_num_sum += 1
                    for idx in top4:
                        if int(idx) in input_word[ii]:
                            continue
                        if int(idx) == int(label[ii][0]):
                            accum_num += 1
                        break
                if step_id % 1 == 0:
                    logger.info("step:%d %d " % (step_id, accum_num))
            acc = 1.0 * accum_num / accum_num_sum
            logger.info("acc:%.3f " % acc)
            infer_result['acc'] = acc
        with open(model_path + '.acc', 'w') as fout:
            fout.write(str(infer_result) + '\n')
        logger.info("Infer Success")

    def run_local(self, params):
        self.inputs = self.input_data(params)
        self.loss = self.net(self.inputs, params)
        self.optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.layers.exponential_decay(
                learning_rate=params.learning_rate,
                decay_steps=params.decay_steps,
                decay_rate=params.decay_rate,
                staircase=True))
        self.optimizer.minimize(self.loss)

        dataset = self.dataset_reader(self.inputs, params)
        file_list = [str(params.train_files_path) + "/%s" % x
                     for x in os.listdir(params.train_files_path)]
        dataset.set_filelist(file_list)
        all_examples = self.get_example_num(file_list)
        logger.info("file list: {}".format(file_list))

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        for epoch in range(params.epochs):
            class fetch_vars(fluid.executor.FetchHandler):
                def handler(self, fetch_target_vars):
                    loss_value = fetch_target_vars[0]
                    logger.info(
                        "epoch -> {}, loss -> {}, at: {}".format(epoch, loss_value, time.ctime()))

            start_time = time.time()
            # Notice: function train_from_dataset does not return fetch value
            # Using fetch_vars to get more information
            exe.train_from_dataset(program=fluid.default_main_program(), dataset=dataset,
                                   fetch_handler=fetch_vars([self.loss.name], 5, True))
            end_time = time.time()
            training_time = float(end_time - start_time)
            speed = float(all_examples) / training_time
            logger.info("epoch: %d finished, using time: %f s ,speed: %f example/s" %
                        (epoch, training_time, speed))
            model_path = str(params.model_path) + \
                '/local_' + '_epoch_' + str(epoch)
            fluid.io.save_persistables(executor=exe, dirname=model_path)
            self.run_infer(params, model_path)

        logger.info("Local train Success!")

    def run_local_cluster(self, params):
        trainers = int(os.getenv("PADDLE_TRAINERS_NUM"))
        pserver_ip_port_list = os.getenv("PADDLE_PSERVERS_IP_PORT_LIST")

        pserver_ports = os.getenv("PADDLE_PORT")
        pserver_ip = os.getenv("POD_IP")
        curr_endpoint = ":".join([pserver_ip, pserver_ports])

        pserver_endpoints = pserver_ip_port_list.split(",")

        current_role = os.getenv("TRAINING_ROLE")
        if current_role == "PSERVER":
            current_role = role_maker.Role.SERVER
            current_id = pserver_endpoints.index(curr_endpoint)
        elif current_role == "TRAINER":
            current_role = role_maker.Role.WORKER
            current_id = int(os.getenv("PADDLE_TRAINER_ID"))

        self.role = role_maker.UserDefinedRoleMaker(
            current_id=current_id,
            role=current_role,
            worker_num=trainers,
            server_endpoints=pserver_endpoints)
        fleet.init(self.role)

        self.strategy = DistributeTranspilerConfig()
        self.strategy.sync_mode = False
        self.strategy.geo_sgd_mode = True
        self.strategy.geo_sgd_need_push_nums = 400
        params.decay_steps = params.decay_steps/self.role.worker_num()

        # step4: Creat network and minimize loss
        self.inputs = self.input_data(params)

        self.loss = self.net(self.inputs, params)

        self.optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.layers.exponential_decay(
                learning_rate=params.learning_rate,
                decay_steps=params.decay_steps,
                decay_rate=params.decay_rate,
                staircase=True))
        self.optimizer = fleet.distributed_optimizer(
            self.optimizer, self.strategy)
        self.optimizer.minimize(self.loss)

        # Step5: According to the parameters-> TRAINING_ROLE, decide which method to run
        if self.role.is_server():
            self.run_pserver(params)
        elif self.role.is_worker():
            self.run_dataset_trainer(params)
        else:
            raise ValueError(
                "Please choice training role for current node : PSERVER / TRAINER")

        logger.info("Distribute train success!")

    def get_example_num(self, file_list):
        count = 0
        for f in file_list:
            last_count = count
            for index, line in enumerate(open(f, 'r')):
                count += 1
            logger.info("file: %s has %s examples" % (f, count-last_count))
        logger.info("Total example: %s" % count)
        return count
