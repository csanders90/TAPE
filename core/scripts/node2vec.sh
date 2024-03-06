python wb_tune.py --cfg core/configs/arxiv_2023/node2vec.yaml --sweep core/configs/arxiv_2023/sweep.yaml

python wb_tune2.py --cfg core/configs/cora/node2vec.yaml --sweep core/configs/cora/sweep.yaml

python wb_tune2.py --cfg core/configs/pubmed/node2vec.yaml --sweep core/configs/pubmed/sweep.yaml



# Today 
python wb_tune2.py --cfg core/configs/cora/node2vec.yaml --sweep core/configs/cora/sweep2.yaml

python wb_tune2.py --cfg core/configs/arxiv_2023/node2vec.yaml --sweep core/configs/arxiv_2023/sweep2.yaml


# Evening
python wb_tune2.py --cfg core/configs/cora/node2vec.yaml --sweep core/configs/cora/sweep3.yaml


python wb_tune2.py --cfg core/configs/arxiv_2023/node2vec.yaml --sweep core/configs/arxiv_2023/sweep3.yaml

