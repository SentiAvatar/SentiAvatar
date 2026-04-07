# Model Checkpoints

请从以下位置下载模型权重，并放置到此目录下：

## 目录结构

```
checkpoints/
├── llm/                              # Qwen2-0.5B SFT (Motion Token Planner)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
├── mask_transformer/                 # Audio-Motion Mask Transformer
│   ├── config.json
│   └── model.safetensors
├── rvqvae/                           # Residual VQ-VAE
│   ├── opt.txt                       # 模型配置
│   └── model/
│       └── epoch_30.pth              # 模型权重
├── face_vqvae/                       # Face VQVAE
│   ├── pytorch_model_face_fad2cl_260116_codesize2048_codelength512.bin
│   ├── mat_final.npy
│   └── mat_final_R_I.npy
├── chinese-hubert-base/              # Chinese HuBERT
│   ├── config.json
│   ├── preprocessor_config.json
│   └── pytorch_model.bin
└── eval_model/                       # ChronAccRet 评测模型
    └── best_model.pt
```
