**aquila**
- `system` 系统级环境变量注入
  - **并行策略参数**
    - `tensor_model_parallel_size` 张量模型并行
    - `pipeline_model_parallel_size`
    - `sequence_parallel`
  - **注意力机制优化项**
    - `reset_position_ids`
    - `reset_attention_mask`
    - `add_qkv_bias`
  - **混合精度训练体系**
    - `fp16`
    - `initial_loss_scale`
    - `min_loss_scale`
    - `attention_softmax_in_fp32`
    - `accumulate_allreduce_grads_in_fp32`
  - **异构计算资源调度**
    - `enable_hetero`
    - `use_partial_reduce_for_shared_embedding`
    - `hetero_pipeline_layer_split`
    - `hetero_process_meshes`
    - `hetero_device_types`
    - `standalone_embedding_stage`
    - `hetero_current_device_type`
  - **工程实践类参数**
    - `no_save_optim`
    - `no_save_rng`
    - `save_interval`
    - `tensorboard_log_interval`
    - `log_interval`
    - `no_log_loss_scale_to_tensorboard`
- `model` 
  - **基础架构参数**
    - `attention_backend`
    - `deterministic_mode`
    - `use_mcore_models`
    - `transformer_impl`
  - **网络结构参数**
    - `num_layers`
    - `hidden_size`
    - `num_attention_heads`
    - `seq_length`
    - `max_position_embeddings`
    - `norm_epsilon`
    - `use_rotary_position_embeddings`
    - `no_position_embedding`
    - `swiglu`
    - `multiple_of`
    - `normalization`
  - **正则化与优化设置**
    - `attention_dropout`
    - `hidden_dropout`
    - `weight_decay`
    - `clip_grad`
  - **训练流程控制**
    - `train_iters`
    - `micro_batch_size`
    - `global_batch_size`
    - `seed`
- **optimizer**
  - `weight_decay`
  - `adam_beta1`
  - `adam_beta2`
  - `lr_scheduler`
    - `lr`
    - `min_lr`
    - `lr_warmup_samples`
    - `lr_warmup_fraction`
    - `lr_decay_style`
- `data`
  - `data_path`
  - `split`
  - `tokenizer`
    - `tokenizer_type`
    - `vocab_file`
    - `merge_file`
    - `special_tokens_file`
    - `vocab_size`
  - ``
```yaml
system:
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 2
  reset_position_ids: True
  reset_attention_mask: True
  add_qkv_bias: True
  sequence_parallel: True
  disable_bias_linear: True
  use_distributed_optimizer: True
  hetero:
    enable_hetero: True
    use_partial_reduce_for_shared_embedding: True
    hetero_pipeline_layer_split: [4, 4]
    hetero_process_meshes: [1,1,1,2,1, 1,1,1,4,1]
    hetero_device_types: ["A800", "A800"]

    standalone_embedding_stage: False
    hetero_current_device_type: "A800"
  precision:
    fp16: True
    initial_loss_scale: 522893
    min_loss_scale: 1.0
    attention_softmax_in_fp32: True
    accumulate_allreduce_grads_in_fp32: True
  logging:
    log_interval: 1
    no_log_loss_scale_to_tensorboard: true
  checkpoint:
    no_save_optim: true
    no_save_rng: true
    save_interval: 100000
    tensorboard_log_interval: 999999

model:
  attention_backend: unfused
  deterministic_mode: true
  use_mcore_models: true
  transformer_impl: transformer_engine
  num_layers: 8
  hidden_size: 512
  num_attention_heads: 8
  seq_length: 1024
  max_position_embeddings: 1024
  norm_epsilon: 1e-5
  use_rotary_position_embeddings: true
  no_position_embedding: true
  swiglu: true
  multiple_of: 256
  normalization: RMSNorm
  # rotary_interleaved_patch: true
  untie_embeddings_and_output_weights: false
  init_method_std: 0.02
  attention_dropout: 0.0
  hidden_dropout: 0.0
  weight_decay: 0.1
  clip_grad: 1.0
  train_iters: 10
  micro_batch_size: 4
  global_batch_size: 1024
  seed: 42

  optimizer:
    weight_decay: 0.1
    adam_beta1: 0.9
    adam_beta2: 0.95
    lr_scheduler:
      lr: 2.0e-5
      min_lr: 2.0e-6
      lr_warmup_samples: 0
      lr_warmup_fraction: 0.01
      lr_decay_style: cosine

data:
  data_path: /home/gitlab-runner/data/pile_wikipedia_demo/pile_wikipedia_demo
  # data_path: /share/project/lizhiyu/FlagScale/build/data/pile_wikipedia_demo
  split: 1
  tokenizer:
    tokenizer_type: AquilaTokenizerFS
    vocab_file: ./examples/aquila/tokenizer/vocab.json
    merge_file: ./examples/aquila/tokenizer/merges.txt
    special_tokens_file: ./examples/aquila/tokenizer/special_tokens.txt
    vocab_size: 100008
```