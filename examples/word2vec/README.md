# 从零开始的分布式Word2Vec
这篇文档，将从零开始，一步步教您如何从搭建`word2vec单机模型`并升级为可以在集群中运行的`CPU分布式模型`。在完成该示例后，您可以入门`PaddlePaddle参数服务器`搭建，了解CPU多线程全异步模式的启用方法，并能够使用`GEO-SGD`模式加速分布式的训练。
## 运行环境检查

- 请确保您的运行环境为`Unbuntu`或`CentOS`
- 请确保您的`PaddlePaddle`版本高于`1.6.0`
- 请确保您的分布式CPU集群支持运行PaddlePaddle
- 请确保您的运行环境中`没有设置http/https代理`

## 示例代码下载
本示例代码运行于Linux环境中，请先安装`git`：https://git-scm.com
，然后在工作目录Clone示例代码：
```bash
git clone https://github.com/PaddlePaddle/Fleet.git
cd Fleet/example/word2vec
```
示例代码位于`Fleet/example/word2vec`

## 数据准备
可以使用一键命令进行数据的下载与预处理：
```bash
sh get_data.sh
```
也可以跟随下述文档，一步步进行数据的准备工作。

### 训练数据下载
在本示例中，Word2Vec模型使用`1 Billion Word Language Model Benchmark`数据,下载地址：
>http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz

在linux环境下执行以下命令进行数据的下载：
```bash
mkdir data
wget http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
tar xzvf 1-billion-word-language-modeling-benchmark-r13output.tar.gz
mv 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/ data/
```
您也可以从国内源上下载数据，速度更快更稳定：
>https://paddlerec.bj.bcebos.com/word2vec/1-billion-word-language-modeling-benchmark-r13output.tar

在linux环境下执行以下命令
```bash
mkdir data
wget https://paddlerec.bj.bcebos.com/word2vec/1-billion-word-language-modeling-benchmark-r13output.tar
tar xvf 1-billion-word-language-modeling-benchmark-r13output.tar
mv 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/ data/
```
### 预测数据下载
```bash
wget https://paddlerec.bj.bcebos.com/word2vec/test_dir.tar
tar -xvf test_dir.tar
mkdir test_data
mv data/test_dir/* ./test_data
```
预测数据是由4词的组合构成，组合中，前两个词之间，及后两个词之间都有紧密的联系，例如:
> Beijing China Tokyo Japan  
> write writes go goes

我们希望训练得到的word2vec模型能够学习到词向量之间的关系，比如首先给出`Beijing China`的示例，再给出`Tokyo`，能够得到`Japan`的结果。以这样的标准，去计算整个模型的评分。

### 数据预处理
下面开始进行数据预处理，主要有两个步骤：1、根据英文语料生成词典；2、根据词典将文本数据转成id，同时进行downsample，按照概率过滤常见词，同时生成word和id映射的文件，名为`词典 + word_to_id`
第一步：根据英文语料生成词典
```bash
python preprocess.py --build_dict --build_dict_corpus_dir data/training-monolingual.tokenized.shuffled --dict_path data/test_build_dict
```
第二步：根据词典将训练文本转成id
```bash
python preprocess.py --filter_corpus --dict_path data/test_build_dict --input_corpus_dir data/training-monolingual.tokenized.shuffled --output_corpus_dir data/convert_text8 --min_count 5 --downsample 0.001
```

为了能够更好的发挥分布式加速性能，我们建议您将数据进行预处理，进行大小的拆分，推荐您将数据集拆分为1024个文件，以**确保每个训练节点的每个线程都能拿到数据文件**。
以下脚本可将数据文件拆分为1024份：
```bash
python preprocess.py --data_resplit --input_corpus_dir=data/convert_text8 --output_corpus_dir=train_data
```
## 训练模型
首先让我们从零开始搭建单机word2vec模型。

### 模型设计及代码
本示例实现了基于skip_gram的Word2Vec模型，模型设计可以参考：
>https://aistudio.baidu.com/aistudio/projectDetail/124377

论文可以参考：
>https://arxiv.org/pdf/1301.3781.pdf

基于PaddlePaddle复现的官方Word2Vec模型可以参考：
>https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/word2vec

请您`配合示例代码阅读以下文档`，本示例的代码位于：
>https://github.com/PaddlePaddle/Fleet/tree/develop/examples/word2vec


### 数据接口
数据接口代码位于`model.py`。

Word2Vec模型有三个输入数据，分别是`input_word`， `true_label`， `neg_label`。`input_word`是当前样本的中心词，`true_label`是窗口内的临近词。`neg_label`是根据自定的负采样率生成的负样本(假设不位于窗口内的词为负样本)。它们的定义如下：
```python
def input_data(self, params):
    input_word = fluid.layers.data(name="input_word", shape=[1], dtype='int64',lod_level=1)
    true_word = fluid.layers.data(name='true_label', shape=[1], dtype='int64',lod_level=1)
    neg_word = fluid.layers.data(name="neg_label", shape=[1], dtype='int64',lod_level=1)
    self.data = [input_word, true_word, neg_word]
    return self.data
```

### 模型组网
了解数据的输入后，让我们开始组网，模型组网代码位于`model.py`。我们的网络希望能够学习到中心词与临近词的相关关系，同时，降低与负样本词的联系。

```python
def net(self, inputs, params):
    init_width = 0.5 / params.embedding_size
    # 根据embedding表——emb，基于input_word的值进行查表
    input_emb = fluid.layers.embedding(
        input=inputs[0],
        is_sparse=params.is_sparse,
        size=[params.dict_size, params.embedding_size],
        param_attr=fluid.ParamAttr(
            name='emb',
            initializer=fluid.initializer.Uniform(-init_width, init_width)))
    # 根据embedding表——emb_w，基于true_word的值进行查表
    true_emb_w = fluid.layers.embedding(
        input=inputs[1],
        is_sparse=params.is_sparse,
        size=[params.dict_size, params.embedding_size],
        param_attr=fluid.ParamAttr(
            name='emb_w', initializer=fluid.initializer.Constant(value=0.0)))
    # 根据embedding表——emb_b，基于true_word的值进行查表
    true_emb_b = fluid.layers.embedding(
        input=inputs[1],
        is_sparse=params.is_sparse,
        size=[params.dict_size, 1],
        param_attr=fluid.ParamAttr(
            name='emb_b', initializer=fluid.initializer.Constant(value=0.0)))

    neg_word_reshape = fluid.layers.reshape(inputs[2], shape=[-1, 1])
    # 负样例不需要梯度下降更新参数
    neg_word_reshape.stop_gradient = True

    # 根据embedding表——emb_w，基于neg_word的值进行查表
    neg_emb_w = fluid.layers.embedding(
        input=neg_word_reshape,
        is_sparse=params.is_sparse,
        size=[params.dict_size, params.embedding_size],
        param_attr=fluid.ParamAttr(
            name='emb_w', learning_rate=1.0))
    neg_emb_w_re = fluid.layers.reshape(
        neg_emb_w, shape=[-1, params.neg_num, params.embedding_size])
    
    # 根据embedding表——emb_b，基于neg_word的值进行查表
    neg_emb_b = fluid.layers.embedding(
        input=neg_word_reshape,
        is_sparse=params.is_sparse,
        size=[params.dict_size, 1],
        param_attr=fluid.ParamAttr(
            name='emb_b', learning_rate=1.0))

    neg_emb_b_vec = fluid.layers.reshape(neg_emb_b, shape=[-1, params.neg_num])

    # wx+b构成true_logits以及neg_logits
    true_logits = fluid.layers.elementwise_add(
        fluid.layers.reduce_sum(
            fluid.layers.elementwise_mul(input_emb, true_emb_w),
            dim=1,
            keep_dim=True),
        true_emb_b)

    input_emb_re = fluid.layers.reshape(
        input_emb, shape=[-1, 1, params.embedding_size])

    neg_matmul = fluid.layers.matmul(
        input_emb_re, neg_emb_w_re, transpose_y=True)
    neg_matmul_re = fluid.layers.reshape(neg_matmul, shape=[-1, params.neg_num])
    neg_logits = fluid.layers.elementwise_add(neg_matmul_re, neg_emb_b_vec)

    # 我们希望正样本的预测概率都为1，而负样本的预测概率都为0
    label_ones = fluid.layers.fill_constant_batch_size_like(
        true_logits, shape=[-1, 1], value=1.0, dtype='float32')
    label_zeros = fluid.layers.fill_constant_batch_size_like(
        true_logits, shape=[-1, params.neg_num], value=0.0, dtype='float32')
    
    # 对正样本及负样本的每个词计算交叉熵损失，正样本的标签为1，负样本的标签为0
    true_xent = fluid.layers.sigmoid_cross_entropy_with_logits(true_logits,
                                                                label_ones)
    neg_xent = fluid.layers.sigmoid_cross_entropy_with_logits(neg_logits,
                                                                label_zeros)
    # 我们希望在正负样本上都有较好的预测表现，需要正负损失之和最小
    cost = fluid.layers.elementwise_add(
        fluid.layers.reduce_sum(
            true_xent, dim=1),
        fluid.layers.reduce_sum(
            neg_xent, dim=1))
    avg_cost = fluid.layers.reduce_mean(cost)
    return avg_cost
```
### 数据读取
在本示例中，数据读取使用dataset API。dataset使用可以查阅:
>[fluid.dataset](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.6/api_cn/dataset_cn.html)

dataset是一种高性能的IO方式，在分布式应用场景中，多线程全异步模式下，使用dataset进行数据读取加速是最佳选择。
dataset读取数据的代码位于`dataset_generator.py`，核心部分如下：
```python
def generate_sample(self, line):
    # 数据读取核心代码
    def data_iter():
        cs = np.array(self.id_frequencys_pow).cumsum()
        neg_array = cs.searchsorted(np.random.sample(neg_num))
        id_ = 0
        word_ids = [w for w in line.split()]
        for idx, target_id in enumerate(word_ids):
            # target_id 当前的中心词，即input_word
            # context_word_dis 获取窗口中的其他词，作为true_label
            context_word_ids = self.get_context_words(
                word_ids, idx)
            for context_id in context_word_ids:
                neg_id = [ int(str(i)) for i in neg_array ]
                output = [('input_word', [int(target_id)]), ('true_label', [int(context_id)]), ('neg_label', neg_id)]
                yield output
                id_ += 1
                # 每个batch内的neg_array一致，该batch结束后，会重新采样一批负样本
                if id_ % self.batch_size == 0:
                    neg_array = cs.searchsorted(np.random.sample(neg_num)) 
    return data_iter
```

下面简要介绍数据IO代码的调试方式，在linux环境下，使用以下命令查看运行结果：

```bash
cat data_file | python dataset_generator.py
```

输出的数据格式如下，依次为：
>input_word:size ; input_word:value ; true_label:size ; input_label:value ;neg_label:size ; neg_word:value ;

上下文窗口默认设置为5，故除input_word外，有1~10个true_label,若有多于10个true_label，可能是叠词的情况。理想的输出如下：
```bash
...

1 406 1 15 5 22 851 202 44666 178398
1 15 1 406 5 22 851 202 44666 178398
1 15 1 527 5 22 851 202 44666 178398
1 527 1 61 5 22 851 202 44666 178398
1 527 1 220 5 22 851 202 44666 178398
1 527 1 12671 5 22 851 202 44666 178398
1 527 1 4777 5 22 851 202 44666 178398
1 527 1 406 5 22 851 202 44666 178398
1 527 1 15 5 22 851 202 44666 178398
1 527 1 955 5 22 851 202 44666 178398
1 955 1 4777 5 22 851 202 44666 178398
1 955 1 406 5 22 851 202 44666 178398
...
```
如何在模型中引入dataset的读取方式呢？示例代码位于`model.py`：
1. 使用dataset，第一步是为dataset设置读取的Variable的格式，在`set_use_var`中添加我们在`数据接口`部分中设置好的数据，该数据是`list[variable]`的形式；
2. 然后我们需要通过`pipe_command`添加读取数据的脚本文件`dataset_generator.py`，dataset类会调用`fluid.DatasetFactory()`其中的`run_from_stdin()`方法进行读取;
3. 读取过程中的线程数由`set_thread()`方法指定，需要说明的是，利用dataset进行模型训练，读取线程与训练时的线程是耦合的，1个读取队列对应1个训练线程，不同线程持有不同文件，这也就是我们在`数据处理`中强调文件数大于线程数的原因所在。
4. 最后，数据输入训练线程的batch_size由`set_batch_size()`方法设置。
```python
def dataset_reader(self, inputs, params):
    dataset = fluid.DatasetFactory().create_dataset()
    # set_use_var的顺序严格要求与读取的顺序一致
    dataset.set_use_var(self.data)
    # 使用pipe command进行数据的高速读取
    pipe_command = "python dataset_generator.py"
    dataset.set_pipe_command(pipe_command)
    # 数据读取的batch_size与训练时保持一致
    dataset.set_batch_size(params.batch_size)
    # 多线程可以充分发挥dataset的速度优势
    # 可以在argument.py中进行线程默认值的修改
    thread_num = int(params.cpu_num)
    dataset.set_thread(thread_num)
    return dataset
```

### 开始训练
下面介绍单机训练的方法，但因word2vec数据较重，单机训练效率较低，在此仅作示例。
在我们完成模型的`input_data`与`net`定义，并实现`数据IO`的代码后，便可以开始进行训练。

以下代码位于`distribute_base.py`中的`run_local`函数
- 首先引入已经定义好的模型，得到损失函数
  ```python
  self.inputs = self.input_data(params)
  self.loss = self.net(self.inputs, params)
  ```
- 引入SGD Optimizer，并添加学习率衰减策略，最小化损失函数
  ```python
  self.optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.layers.exponential_decay(
                learning_rate=params.learning_rate,
                decay_steps=params.decay_steps,
                decay_rate=params.decay_rate,
                staircase=True))
  self.optimizer.minimize(self.loss)
  ```
- 引入定义好的dataset数据读取模块，并设置文件路径
  ```python
  dataset = self.dataset_reader(self.inputs, params)
  file_list = [str(params.train_files_path) + "/%s" % x
                for x in os.listdir(params.train_files_path)]
  dataset.set_filelist(file_list)
  ```
- 设置训练程序(program)的执行位置，并初始化参数
  ```python
  exe = fluid.Executor(fluid.CPUPlace())
  exe.run(fluid.default_startup_program())
  ```
- 执行训练。
  - 因为dataset设计初衷是保证高速，所以运行于程序底层，与paddlepaddle传统的`feed={dict}`方法不一致，不支持直接通过`train_from_dataset`的返回值监控当前训练的细节，比如loss的变化，但我们可以通过1.6新增的`fetch_handler`方法创建一个新的线程，监听训练过程，不影响训练的效率。该方法需要继承`fluid.executor.FetchHandler`类中的`handler`方法实现一个监听函数。`fetch_target_vars`是一个list，由我们自行指定哪些变量的值需要被监控。
  - 在`exe.train_from_dataset`方法中，指定`fetch_handler`为我们实现的监听函数。可以配置3个超参：
    1. 第一个是`fetch_var_list`，添加我们想要获取的变量的名称，示例中，我们指定为`[self.loss.name]`
    2. 第二个是监听函数的更新频率，单位是s，示例中我们设置为5s更新一次。
    3. 第三个是我们获取的变量的数据类型，若想获得常用的`numpy.ndarray`的格式，则设置为`True`；若想获得`Tensor`，则设置为`False`
   
  ```python
  for epoch in range(params.epochs):
    # 实现监听函数
    class fetch_vars(fluid.executor.FetchHandler):
        def handler(self, fetch_target_vars):
            loss_value = fetch_target_vars[0]
            logger.info(
                "epoch -> {}, loss -> {}, at: {}".format(epoch, loss_value, time.ctime()))
    # 开始训练
    start_time = time.time()
    # Notice: function train_from_dataset does not return fetch value
    # Using fetch_handler to get more information
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
  ```
完成以上步骤，便可以开始进行单机的训练。在示例代码中，运行单机训练，需要在代码目录下输入命令：
```bash
sh local_cluster.sh local
```

日志文件可以在./log/local_training.log中查阅，理想的输出为：
```bash
2019-10-24 08:53:30,053 - INFO - Local train start
2019-10-24 08:53:30,095 - INFO - file: train_data/convert_text8_0 has 1 examples
2019-10-24 08:53:30,096 - INFO - file: train_data/convert_text8_1 has 1 examples
2019-10-24 08:53:30,096 - INFO - Total example: 2
2019-10-24 08:53:30,096 - INFO - file list: ['train_data/convert_text8_0', 'train_data/convert_text8_1']
2019-10-24 08:53:35,156 - INFO - epoch -> 0, loss -> [3.8394504], at: Thu Oct 24 08:53:35 2019
2019-10-24 08:53:40,162 - INFO - epoch -> 0, loss -> [3.7490945], at: Thu Oct 24 08:53:40 2019
2019-10-24 08:53:45,167 - INFO - epoch -> 0, loss -> [3.7623515], at: Thu Oct 24 08:53:45 2019
2019-10-24 08:53:50,172 - INFO - epoch -> 0, loss -> [3.7113543], at: Thu Oct 24 08:53:50 2019
2019-10-24 08:53:55,177 - INFO - epoch -> 0, loss -> [3.6614287], at: Thu Oct 24 08:53:55 2019
2019-10-24 08:54:00,182 - INFO - epoch -> 0, loss -> [3.6755407], at: Thu Oct 24 08:54:00 2019
2019-10-24 08:54:05,187 - INFO - epoch -> 0, loss -> [3.5342407], at: Thu Oct 24 08:54:05 2019
2019-10-24 08:54:10,084 - INFO - epoch: 0 finished, using time: 39.937401 s ,speed: 0.050078 example/s
2019-10-24 08:54:10,126 - INFO - Local train Success!
```
## 分布式训练——Fluid-GEO-SGD模式
PaddlePaddle在1.5.0之后新增了`Fleet`的高级分布式API，可以只需数行代码便可将单机模型转换为分布式模型,使用Fleet构建基于参数服务器`Parameter Server`架构的CPU分布式训练架构，可以在CPU集群上加速我们的训练，参数服务器的介绍可以参考：
>[分布式CPU训练优秀实践](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_usage/best_practice/cpu_train_best_practice.html)

传统的参数服务器各个Trainer与Pserver之间的参数交互使用`同步模式（Sync）`，这种模式需要Pserver每个batch都等待所有节点训练完成才会进行下一次训练，可以保证效果与单机对齐，但速度非常慢。Fluid基于简化版的Downpour-SGD实现了`全异步模式（Async）`，这种模式可以做到多线程全异步全局无锁更新，但其限制在于仍然是每个batch通信一次，速度仍有提升的潜力。全异步模式的训练方法，可以参考：
>[分布式CPU训练优秀实践](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_usage/best_practice/cpu_train_best_practice.html)

PaddlePaddle在1.6.0之后新增了`GEO-SGD模式`，这种模式也是多线程全异步全局无锁的高速模式，支持每个节点在本地训练一定步长后，再与Pserver通信，进行全局参数的更新，显著提升了训练速度，特别是对于Word2Vec这类有大量稀疏参数的模型，提速会更加明显。接下来，本示例便展示如何从单机模型构建一个`Fleet-ParameterServer-GeoSgd模式`的CPU分布式训练。

参数服务器架构中，有两种角色，分别是：Pserver——全局参数的统筹者；Trainer——利用局部数据更新参数。在集群中运行时，一般是通过配置每个节点的环境变量来确定该节点所扮演的角色。我们首先构写准备步骤的代码，以下代码位于`distribute_base.py`中的`runtime_main`函数。
- 通过环境变量确定当前节点的角色：Pserver or Trainer ？
  
  Fleet提供了非常易用的API帮助我们完成这一步骤，仅需两行代码！使用`PaddleCloudRoleMaker`构建节点角色，其会自动的从当前节点的环境变量中获取所需配置（包括通信所需IP、端口，该节点的角色，以及其他节点的信息），并完成构建的工作。随后，我们使用`fleet.init()`方法初始化该节点的环境配置。
  ```python
  self.role = role_maker.PaddleCloudRoleMaker()
  fleet.init(self.role)
  ```
- 确定分布式训练的模式，配置`分布式strategy`，我们选择`Fleet-ParameterServer-GeoSgd模式`，需要进行如下配置。将sync_mode设置为False，便默认开启`全异步模式（Async）`，在此基础上，设置geo_sgd_mode，开启`GeoSgd模式`，我们可以通过设置geo_sgd_need_push_nums指定本地训练的步长，示例中设置为训练400个batch（16线程，每个线程只需训练25batch）再进行全局通信。
  ```python
  from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig

  self.strategy = DistributeTranspilerConfig()
  self.strategy.sync_mode = False            
  self.strategy.geo_sgd_mode = True
  self.strategy.geo_sgd_need_push_nums = 400
  ```

- 接下来我们如单机训练一样，构建网络的输入与损失函数
  ```python
  self.inputs = self.input_data(params)
  self.loss = self.net(self.inputs, params)
  ```
- 配置optimizer，并最小化损失函数。注意，此处需要将optimizer转化为分布式的optimizer，并将刚才定义的`分布式strategy`作为参数传入。
  ```python
  self.optimizer = fluid.optimizer.SGD(
                learning_rate=fluid.layers.exponential_decay(
                    learning_rate=params.learning_rate,
                    decay_steps=params.decay_steps,
                    decay_rate=params.decay_rate,
                    staircase=True))
  self.optimizer = fleet.distributed_optimizer(self.optimizer, self.strategy)
  self.optimizer.minimize(self.loss)
  ```
- 根据当前节点的role，我们分别运行Pserver与Trainer的流程。
  ```python
  if self.role.is_server():
      self.run_pserver(params)
  elif self.role.is_worker():
      self.run_dataset_trainer(params)
  ```
- Pserver的流程非常简单，只有两行代码，我们需要先初始化server，再开始运行。
  ```python
  def run_pserver(self, params):
      fleet.init_server()
      fleet.run_server()
  ```
- Traine的流程与单机非常相似，只需改动数行代码，首先我们初始化worker节点，并如单机一样构建执行器，进行参数的初始化。注意，此处传入的program是`fleet.startup_program`.
  ```python
  fleet.init_worker()
  exe = fluid.Executor(fluid.CPUPlace())
  exe.run(fleet.startup_program)
  ```
- 引入定义好的dataset数据读取模块，并设置文件路径。在分配文件时，我们应该尽量做到每个节点的文件数量均匀且各不相同。在本地模拟分布式时，可以通过`fleet.split_files()`完成这一步骤。
  ```python
  dataset = self.dataset_reader(self.inputs, params)
  file_list = [str(params.train_files_path) + "/%s" % x
               for x in os.listdir(params.train_files_path)]
  dataset.set_filelist(file_list)
  ```
- 开始训练，此处与单机基本一致。注意：需要将`train_from_dataset()`中的program替换为`fleet.main_program`
  ```python
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
      logger.info("epoch: %d finished, using time: %f s ,speed: %f example/s" %
                (epoch, training_time, speed))

      if self.role.is_first_worker() and params.test:
          model_path = str(params.model_path) + '/trainer_' + \
              str(self.role.worker_index()) + '_epoch_' + str(epoch)
          fleet.save_persistables(executor=exe, dirname=model_path)
  ```
完成以上步骤，便可以进行基于GEO-SGD模式的CPU分布式训练。在示例代码中，我们配置一个2x2的参数服务器架构，即两个pserver，两个trainer，进行1个epoch的训练。
直接使用示例代码运行本地多进程模拟分布式的命令如下：
```bash
sh local_cluster.sh local_cluster
```
日志输出如下：
```bash
2019-10-24 08:38:12,669 - INFO - Local cluster train start
I1024 08:38:12.733052 26181 communicator_py.cc:52] using geo sgd communicator
I1024 08:38:12.733481 26181 communicator.cc:387] communicator_independent_recv_thread: 1
I1024 08:38:12.733501 26181 communicator.cc:389] communicator_send_queue_size: 1
I1024 08:38:12.733510 26181 communicator.cc:391] communicator_min_send_grad_num_before_recv: 20
I1024 08:38:12.733517 26181 communicator.cc:393] communicator_thread_pool_size: 5
I1024 08:38:12.733525 26181 communicator.cc:395] communicator_send_wait_times: 5
I1024 08:38:12.733533 26181 communicator.cc:397] communicator_max_merge_var_num: 20
I1024 08:38:12.733541 26181 communicator.cc:399] communicator_fake_rpc: 0
I1024 08:38:12.733548 26181 communicator.cc:400] communicator_merge_sparse_grad: 1
I1024 08:38:12.733556 26181 communicator.cc:402] Trainer nums: 2
I1024 08:38:12.733564 26181 communicator.cc:403] geo_sgd_push_before_local_train_nums: 400
I1024 08:38:12.733572 26181 communicator.cc:404] communicator_merge_sparse_bucket 2000
I1024 08:38:12.733930 26181 communicator.cc:460] Geo Sgd Communicator start
I1024 08:38:12.733947 26181 communicator.cc:464] start send thread
I1024 08:38:12.734040 26441 communicator.cc:543] SendThread start!
I1024 08:38:12.786202 26181 rpc_client.h:106] init rpc client with trainer_id 1
2019-10-24 08:38:12,892 - INFO - file list: ['train_data/convert_text8_1']
2019-10-24 08:38:12,893 - INFO - file: train_data/convert_text8_1 has 1 examples
2019-10-24 08:38:12,893 - INFO - Total example: 1
2019-10-24 08:38:17,903 - INFO - epoch -> 0, loss -> [3.9996336], at: Thu Oct 24 08:38:17 2019
2019-10-24 08:38:22,909 - INFO - epoch -> 0, loss -> [3.8491988], at: Thu Oct 24 08:38:22 2019
2019-10-24 08:38:27,914 - INFO - epoch -> 0, loss -> [3.8310702], at: Thu Oct 24 08:38:27 2019
2019-10-24 08:38:32,919 - INFO - epoch -> 0, loss -> [3.781963], at: Thu Oct 24 08:38:32 2019
2019-10-24 08:38:37,924 - INFO - epoch -> 0, loss -> [3.8192966], at: Thu Oct 24 08:38:37 2019
2019-10-24 08:38:42,929 - INFO - epoch -> 0, loss -> [3.7822595], at: Thu Oct 24 08:38:42 2019
2019-10-24 08:38:47,934 - INFO - epoch -> 0, loss -> [3.7435296], at: Thu Oct 24 08:38:47 2019
2019-10-24 08:38:52,019 - INFO - epoch: 0 finished, using time: 39.125053 s ,speed: 0.025559 example/s
2019-10-24 08:38:52,019 - INFO - Train Success!
I1024 08:38:52.019517 26181 communicator.cc:473] Geo Sgd Communicator stop
I1024 08:38:52.072788 26181 communicator.cc:484] Geo Sgd Communicator stop done
2019-10-24 08:38:52,073 - INFO - Distribute train success!
```
注意：本地模拟分布式仅用于熟悉Paddle框架及分布式的使用，并不建议在单机环境下利用分布式模拟来训练模型，会存在很严重的性能与通信瓶颈，多个进程抢占计算资源。

## 模型保存
模型保存的使用代码，在上文中已有展现，现进行详细介绍。有关于PaddlePaddle模型保存及加载的使用示例可以参考：
>[模型/变量的保存、载入与增量训练](https://www.paddlepaddle.org.cn/documentation/docs/zh/user_guides/howto/training/save_load_variables.html)

分布式的模型保存与单机模型保存有细微差别，具体如下：
- 单机模型保存
  ```python
  fluid.io.save_persistables(executor=exe, dirname=model_path)
  ```
- 分布式模型保存
  ```python
  fleet.save_persistables(executor=exe, dirname=model_path)
  ```
单机模型使用`fluid.io`接口进行模型的保存，而分布式使用`fleet`接口保存模型。区别在于单机模型时仅保存本地内存中的参数，而分布式模型保存，会从各个Pserver上拉取全局的最新的参数保存下来，全局参数才是准确的训练结果，因此在分布式训练中请使用`fleet`接口保存模型。

在分布式框架中，我们偏爱使用`save_persistables`的方式保存模型中的模型参数，原因在于使用`save_inference_model`占用内存更多，且包含分布式专用的流程与参数，直接使用保存的模型进行单机预测会存在不确定风险。而使用`save_persistables`可以精确保存训练参数，在进行推理时效果更稳定。
## 模型评估
word2vec模型的评价方法在数据预处理时已介绍基本思路，代码实现在`distribute_base.py`的`run_infer()`函数中，我们只需配置模型保存的位置即可加载参数进行模型评估，评估的效果用acc表示。默认会在local模式训练完成后自动进行预测。

单独开启模型预测的命令示例如下，请注意调整为自己模型的保存地址
```bash
sh local_cluster.sh infer model/local__epoch_0
```

## 调试及优化
`Fleet-ParameterServer-GeoSgd模式`中，我们需要关注两方面的调优：速度与效果。以下参数会影响到`GEO-SGD`模式的表现。
- 首先是线程数

  在程序运行时，指定训练所使用的线程数n，基于训练节点CPU所具有的核心数调整。在我们的benchmark测试中，我们设置为16，您可以根据具体环境进行设置。通常来说，线程数越高，训练速度越快。在调整线程数时，您需要关注节点上的文件数是否大于线程数，若文件数少于线程数，则不能如预想的提升速度。同时，我们推荐文件数可以整除线程数，这样可以发挥dataset的最佳性能。
  ```bash
  export CPU_NUM=n
  ``` 
- 然后是本地训练的步长
  
  训练节点在本地训练的轮数越多，则通信耗时占整体耗时会显著降低。频繁的全局参数交互，是有利于各个节点掌握其他节点参数信息，并避免陷入局部最优的。因此，步长`geo_sgd_need_push_num`是一个权衡效果与速度的参数。基于经验值，我们推荐每个线程训练25个batch的步长后进行通信，比如benchmark中，16线程训练，因此设置`geo_sgd_need_push_num`为400。
  ```python
  DistributeTranspilerConfig().geo_sgd_need_push_num = n
  ```
- 调整环境变量
  
  在大型模型的训练中，有很多使用了`embedding layer`的稀疏参数，这类参数占据着通信耗时的主要矛盾，因此我们可以指定处理稀疏参数通信的线程数`FLAGS_communicator_thread_pool_size`：
  ```bash
  export FLAGS_communicator_thread_pool_size=n
  ```
  线程池大小n决定了我们能最多同时启用多少线程来收发稀疏参数，增加该值，可以显著提升模型训练速度。但同时，不能够无限增加该变量，取决于机器的配置与网络带宽。在`GEO-SGD`的训练中，增速的上限为节点数 * 稀疏参数表的数量，超过该值后收益递减。
  同时还有一个环境变量`FLAGS_rpc_retry_times`，该变量适用于网络波动明显，通信不稳定的情形。设置该值，可以在一定程度上保证通信及效果的稳定，我们推荐在`GEO-SGD`模式下设置为3
  ```bash
  export FLAGS_rpc_retry_times=3
  ```
- 全局优化参数的调整
  由于`GEO-SGD`是在各个节点上独立训练，使用参数增量进行全局参数的更新，因此一些全局的优化参数需要在该模式下进行调整。比如学习率衰减策略中，我们可以设置`decay_step`来进行学习率的固定步长后衰减，但此处的decay_step为全局参数，本地的样本量不足以迭代预想次数，不能如预期一样衰减到目标学习率。因此，我们需要对decay_step进行同比缩放，除以节点数，保证学习率的正常衰减：
  ```python
  decay_step = decay_step / trainer_nums
  ```
## benchmark及效果复现
PaddlePaddle分布式模型在word2vec下测试得到的benchmark数据，包括速度、效果及复现方法，可以查阅：
>[PaddlePaddle-Fleet-Word2Vec Benchmark](https://github.com/PaddlePaddle/Fleet/tree/develop/benchmark/ps/distribute_word2vec/paddle)