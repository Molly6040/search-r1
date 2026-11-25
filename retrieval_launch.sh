file_path=/cache/h30082669/searchr1/wiki-18
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=/cache/h30082669/searchr1/e5-base-v2
# 注: 端口 8010 避开 Tree-GRPO 检索服务(8000); 不加 --faiss_gpu,
# 训练占卡时单卡剩余显存放不下 64.5GB fp32 索引, 用 CPU 检索(本机 2TB 内存)
python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --port 8010
